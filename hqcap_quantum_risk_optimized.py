#!/usr/bin/env python3
"""
HQCAP - Hybrid Quantum-Classical Actuarial Pipeline (Topic 2)
=============================================================
Optimized Python Implementation grounded in Topic 2 Mathematical Specification:
1. Phase 1: Angle Embedding + Variational Quantum Circuit (VQC) Severity Model (SPSA & Adam+Parameter Shift)
2. Phase 2: Quantum Amplitude Estimation (QAE) Tail-Risk Engine & Grover Operator Verification
3. Phase 3: NISQ Error Mitigation (ZNE with Richardson Extrapolation, PEC, REM Constrained Least Squares)
4. Phase 4: Downstream Actuarial Reserving (BEL, VaR99.5%, TVaR99.5%, Solvency II SCR)
5. Phase 5: Empirical Benchmarking, Convergence Scaling Studies & Output Generation

Pure NumPy/SciPy implementation: Quantum simulator, oracle emulator, and error mitigators are built
from scratch to ensure 100% mathematical auditability and zero external quantum SDK dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.optimize import nnls


# ======================================================================================
# 0. CONFIGURATION (Default Parameters set per Methodology Table 2)
# ======================================================================================
@dataclass
class Config:
    seed: int = 7
    n_qubits: int = 4                 # 4..8 feature qubits (AngleEmbedding, Eq. 1)
    n_layers: int = 3                 # entangling layers L (Eq. 2)
    n_policyholders: int = 50_000
    n_train: int = 2_000
    n_val: int = 500
    shots: int = 10_000               # N_shots per circuit execution
    spsa_iters: int = 200
    adam_iters: int = 200
    adam_lr: float = 0.05
    batch: int = 128
    # Hardware noise calibration (depolarising strength per touched qubit after each gate)
    lam1: float = 0.001
    lam2: float = 0.012
    ro_e01: float = 0.015             # P(read 1 | prepared 0)
    ro_e10: float = 0.030             # P(read 0 | prepared 1)
    zne_scales: tuple = (1, 3, 5)     # odd global-folding factors (Eq. 8)
    n_qem_test: int = 24              # test inputs for QEM benchmark
    # Tail-risk loss discretization
    n_loss_qubits: int = 8            # loss grid = 2^8 basis states (Eq. 5)
    alpha: float = 0.995              # Solvency II confidence level
    ref_samples: int = 4_000_000
    tau_sys: float = 0.08             # lognormal systemic volatility
    p_cat: float = 0.03               # catastrophe (extreme weather) probability
    pareto_alpha: float = 2.2
    pareto_xm: float = 0.15
    pmf_shots: int = 400_000
    # QAE parameters
    qae_eps: float = 0.002            # depolarising error per application of oracle A
    scaling_budgets: tuple = (256, 1024, 4096, 16384, 65536)
    scaling_reps: int = 200
    risk_budget: int = 110_000        # total oracle calls for QAE risk metrics
    risk_reps: int = 30
    out_dir: str = "outputs"


# ======================================================================================
# 1. DATA PIPELINE (MEPS / MIMIC-IV Loader + Synthetic Heavy-Tailed Claims Generator)
# ======================================================================================
FEATURES = ["age", "prior_claims", "weather_idx", "chronic", "bmi", "income_idx",
            "region_risk", "smoker"]


def load_claims_csv(path: str, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    """Load real claims CSV (e.g. MEPS full-year file). Removes non-positive expenditure."""
    df = pd.read_csv(path)[list(feature_cols) + [target_col]].dropna()
    df = df[df[target_col] > 0].rename(columns={target_col: "expenditure"})
    return df.reset_index(drop=True)


def make_claims(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """MEPS-like synthetic panel: Log-normal body with heavy-tailed catastrophe contamination."""
    age = rng.uniform(18, 90, n)
    prior = rng.poisson(1.2, n)
    weather = rng.beta(2, 3, n)                                     # extreme-weather exposure
    chronic = rng.poisson(1.0 + (age - 18) / 60, n)
    bmi = rng.normal(27, 5, n)
    income = rng.lognormal(0, 0.5, n)
    region = rng.uniform(0, 1, n)
    smoker = rng.binomial(1, 0.2, n)
    mu = (6.0 + 0.02 * (age - 50) + 0.22 * prior + 0.8 * weather * (1 + 0.02 * (age - 50))
          + 0.20 * chronic + 0.015 * (bmi - 27) + 0.30 * smoker + 0.30 * region
          - 0.10 * np.log(income) + 0.05 * prior * chronic)
    scale = np.where(rng.random(n) < 0.05, 1.6, 0.6)                # 5% wild-noise policies
    y = np.exp(mu + scale * rng.standard_normal(n))
    return pd.DataFrame(dict(zip(FEATURES, [age, prior, weather, chronic, bmi, income,
                                            region, smoker])) | {"expenditure": y})


class Scaler:
    """Feature normalization into rotation angles [0, pi]; log-target into [-1, 1] (<Z> range)."""

    def fit(self, X: np.ndarray, logy: np.ndarray) -> Scaler:
        self.lo, self.hi = X.min(0), X.max(0)
        self.m, self.s = logy.mean(), logy.std()
        return self

    def x(self, X: np.ndarray) -> np.ndarray:
        return np.pi * np.clip((X - self.lo) / (self.hi - self.lo + 1e-12), 0, 1)

    def y(self, logy: np.ndarray) -> np.ndarray:
        return np.clip((logy - self.m) / (2.5 * self.s), -1, 1)

    def inv_y(self, ys: np.ndarray) -> np.ndarray:
        return self.m + 2.5 * self.s * ys


# ======================================================================================
# 2. QUANTUM SIMULATOR (Batched State Vector Simulator; Qubit 0 = MSB)
# ======================================================================================
def rot_matrix(kind: str, ang: np.ndarray) -> np.ndarray:
    ang = np.asarray(ang, float)
    c, s = np.cos(ang / 2), np.sin(ang / 2)
    U = np.zeros(ang.shape + (2, 2), complex)
    if kind == "RX":
        U[..., 0, 0] = U[..., 1, 1] = c
        U[..., 0, 1] = U[..., 1, 0] = -1j * s
    elif kind == "RY":
        U[..., 0, 0] = U[..., 1, 1] = c
        U[..., 0, 1], U[..., 1, 0] = -s, s
    else:                                                           # RZ
        U[..., 0, 0], U[..., 1, 1] = np.exp(-1j * ang / 2), np.exp(1j * ang / 2)
    return U


class QSim:
    def __init__(self, n: int):
        self.n, self.dim = n, 2 ** n
        idx = np.arange(self.dim)
        self.bit = [(idx >> (n - 1 - q)) & 1 for q in range(n)]
        self.xperm = [idx ^ (1 << (n - 1 - q)) for q in range(n)]
        self.zsign = [1 - 2 * b for b in self.bit]
        self.cperm = {}

    def zeros(self, B: int) -> np.ndarray:
        psi = np.zeros((B, self.dim), complex)
        psi[:, 0] = 1.0
        return psi

    def apply_1q(self, psi: np.ndarray, U: np.ndarray, q: int) -> np.ndarray:
        B = psi.shape[0]
        s = psi.reshape(B, 2 ** q, 2, 2 ** (self.n - q - 1))
        out = np.einsum("ij,bkjl->bkil" if U.ndim == 2 else "bij,bkjl->bkil", U, s)
        return out.reshape(B, self.dim)

    def cnot(self, psi: np.ndarray, c: int, t: int) -> np.ndarray:
        if (c, t) not in self.cperm:
            idx = np.arange(self.dim)
            self.cperm[(c, t)] = np.where(self.bit[c] == 1, idx ^ (1 << (self.n - 1 - t)), idx)
        return psi[:, self.cperm[(c, t)]]

    def pauli(self, psi: np.ndarray, q: int, which: np.ndarray) -> np.ndarray:
        """Apply Pauli gates {0:I, 1:X, 2:Y, 3:Z} trajectory-wise on qubit q."""
        out = psi.copy()
        for t in (1, 2, 3):
            idx = np.flatnonzero(which == t)
            if idx.size:
                blk = out[idx]
                if t in (2, 3):
                    blk = blk * self.zsign[q]
                if t in (1, 2):
                    blk = blk[:, self.xperm[q]]
                out[idx] = blk
        return out

    def run_ideal(self, gates: list, B: int) -> np.ndarray:
        psi = self.zeros(B)
        for kind, a, b in gates:
            psi = self.cnot(psi, a, b) if kind == "CNOT" else self.apply_1q(
                psi, rot_matrix(kind, b), a)
        return psi

    def z_expval(self, psi: np.ndarray) -> np.ndarray:
        """Pauli-Z expectation value on readout qubit 0 (Eq. 3)."""
        return (np.abs(psi) ** 2) @ self.zsign[0].astype(float)


def build_circuit(x: np.ndarray, theta: np.ndarray, n: int, L: int) -> list:
    """Eqs. 1-2. Angle embedding + parameterized layers RY(theta)RZ(phi) with CNOT chain."""
    g = [("RX", j, x[..., j]) for j in range(n)]
    for l in range(L):
        for j in range(n):
            g += [("RZ", j, theta[l, j, 1]), ("RY", j, theta[l, j, 0])]
        g += [("CNOT", j, j + 1) for j in range(n - 1)]
    return g


def fold_circuit(gates: list, s: int) -> list:
    r"""Eq. 8: Global unitary folding U(s) = U (U^\dagger U)^((s-1)/2)."""
    inv = [(k, a, (-b if k != "CNOT" else b)) for k, a, b in reversed(gates)]
    return gates + (inv + gates) * ((s - 1) // 2)


# ======================================================================================
# 3. VQC TRAINING (Eqs. 3-4: SPSA & Adam with Parameter-Shift Gradients)
# ======================================================================================
def vqc_predict(sim: QSim, Xa: np.ndarray, theta: np.ndarray, L: int) -> np.ndarray:
    return sim.z_expval(sim.run_ideal(build_circuit(Xa, theta, sim.n, L), len(Xa)))


def mse(sim: QSim, Xa: np.ndarray, y: np.ndarray, theta: np.ndarray, L: int) -> float:
    return float(np.mean((vqc_predict(sim, Xa, theta, L) - y) ** 2))


def train_spsa(sim: QSim, Xa: np.ndarray, y: np.ndarray, cfg: Config, rng: np.random.Generator,
               shot_noise: bool = True) -> tuple[np.ndarray, list[float]]:
    """Two-measurement SPSA optimization (Spall gain sequences)."""
    L, n = cfg.n_layers, cfg.n_qubits
    theta = rng.uniform(-0.1, 0.1, (L, n, 2))
    A, alpha, gam, c = 20.0, 0.602, 0.101, 0.15

    def noisy_loss(th, idx):
        f = vqc_predict(sim, Xa[idx], th, L)
        if shot_noise:
            f = f + rng.standard_normal(len(f)) * np.sqrt((1 - f ** 2) / cfg.shots)
        return float(np.mean((f - y[idx]) ** 2))

    def grad(th, k, idx):
        ck = c / (k + 1) ** gam
        d = rng.choice([-1.0, 1.0], th.shape)
        return (noisy_loss(th + ck * d, idx) - noisy_loss(th - ck * d, idx)) / (2 * ck) * (1 / d)

    g0 = np.mean([np.abs(grad(theta, 0, rng.choice(len(Xa), cfg.batch))).mean()
                  for _ in range(10)])
    a = 0.2 * (A + 1) ** alpha / (g0 + 1e-12)
    hist = []
    for k in range(cfg.spsa_iters):
        idx = rng.choice(len(Xa), cfg.batch, replace=False)
        theta = theta - a / (k + 1 + A) ** alpha * grad(theta, k, idx)
        if k % 10 == 0:
            hist.append(mse(sim, Xa, y, theta, L))
    return theta, hist


def train_adam_ps(sim: QSim, Xa: np.ndarray, y: np.ndarray, cfg: Config,
                    rng: np.random.Generator) -> tuple[np.ndarray, list[float]]:
    """Adam optimizer with exact parameter-shift gradients."""
    L, n = cfg.n_layers, cfg.n_qubits
    theta = rng.uniform(-0.1, 0.1, (L, n, 2))
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    for k in range(1, cfg.adam_iters + 1):
        idx = rng.choice(len(Xa), cfg.batch, replace=False)
        xb, yb = Xa[idx], y[idx]
        f = vqc_predict(sim, xb, theta, L)
        g = np.zeros_like(theta)
        for pos in np.ndindex(theta.shape):
            tp, tm = theta.copy(), theta.copy()
            tp[pos] += np.pi / 2
            tm[pos] -= np.pi / 2
            dfd = 0.5 * (vqc_predict(sim, xb, tp, L) - vqc_predict(sim, xb, tm, L))
            g[pos] = np.mean(2 * (f - yb) * dfd)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        theta = theta - cfg.adam_lr * (m / (1 - b1 ** k)) / (np.sqrt(v / (1 - b2 ** k)) + eps)
        if k % 10 == 0:
            hist.append(mse(sim, Xa, y, theta, L))
    return theta, hist


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0):
    Z = np.c_[np.ones(len(X)), X]
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y)
    return lambda Xn: np.c_[np.ones(len(Xn)), Xn] @ w


def r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


# ======================================================================================
# 4. NISQ ERROR MITIGATION PROTOCOLS (Eqs. 8-12: ZNE, PEC, REM)
# ======================================================================================
def run_noisy(sim: QSim, gates: list, B: int, cfg: Config, rng: np.random.Generator,
              pec: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
    """Gate-level depolarising Pauli noise trajectories and PEC quasi-probability sampling."""
    psi, sign, gamma = sim.zeros(B), np.ones(B), 1.0
    for kind, a, b in gates:
        if kind == "CNOT":
            psi, locs = sim.cnot(psi, a, b), [(a, cfg.lam2), (b, cfg.lam2)]
        else:
            psi, locs = sim.apply_1q(psi, rot_matrix(kind, b), a), [(a, cfg.lam1)]
        for q, lam in locs:
            err = rng.random(B) < 0.75 * lam
            if err.any():
                w = np.zeros(B, int)
                w[err] = rng.integers(1, 4, err.sum())
                psi = sim.pauli(psi, q, w)
            if pec:
                ins = rng.random(B) < lam / (1 + lam)
                if ins.any():
                    w = np.zeros(B, int)
                    w[ins] = rng.integers(0, 4, ins.sum())
                    psi = sim.pauli(psi, q, w)
                    sign[ins] *= -1
                gamma *= (1 + lam) / (1 - lam)
    return psi, sign, gamma


def measure_z(sim: QSim, psi: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Single-shot readout with asymmetric readout errors e01 and e10."""
    p_plus = (np.abs(psi) ** 2) @ (sim.bit[0] == 0).astype(float)
    o = np.where(rng.random(len(psi)) < p_plus, 1.0, -1.0)
    flip = np.where(o > 0, rng.random(len(o)) < cfg.ro_e01, rng.random(len(o)) < cfg.ro_e10)
    return np.where(flip, -o, o)


def rem_expectation(z_meas: float, cfg: Config) -> float:
    """Exact inverse 2x2 confusion matrix calibration for readout error (Eqs. 11-12)."""
    return (z_meas - (cfg.ro_e10 - cfg.ro_e01)) / (1 - cfg.ro_e01 - cfg.ro_e10)


def richardson(scales: tuple, vals: list) -> tuple[float, np.ndarray]:
    """Richardson extrapolation weights and zero-noise estimate (Eq. 9)."""
    s = np.asarray(scales, float)
    w = [np.prod([-s[j] / (s[i] - s[j]) for j in range(len(s)) if j != i]) for i in range(len(s))]
    return float(np.dot(w, vals)), np.asarray(w)


def qem_expectation(sim: QSim, x: np.ndarray, theta: np.ndarray, cfg: Config,
                    rng: np.random.Generator, mode: str) -> float:
    g = build_circuit(x, theta, cfg.n_qubits, cfg.n_layers)
    B = cfg.shots
    if mode == "raw":
        return float(measure_z(sim, run_noisy(sim, g, B, cfg, rng)[0], cfg, rng).mean())
    if mode == "rem":
        return rem_expectation(measure_z(sim, run_noisy(sim, g, B, cfg, rng)[0], cfg, rng).mean(), cfg)
    if mode == "zne":
        vals = [rem_expectation(measure_z(sim, run_noisy(sim, fold_circuit(g, s), B, cfg, rng)[0],
                                          cfg, rng).mean(), cfg) for s in cfg.zne_scales]
        return richardson(cfg.zne_scales, vals)[0]
    if mode == "pec":
        psi, sign, gamma = run_noisy(sim, g, B, cfg, rng, pec=True)
        return rem_expectation(float(np.mean(gamma * sign * measure_z(sim, psi, cfg, rng))), cfg)
    raise ValueError(mode)


def qem_benchmark(sim: QSim, Xtest: np.ndarray, theta: np.ndarray, cfg: Config,
                  rng: np.random.Generator) -> pd.DataFrame:
    rows = {m: [] for m in ("raw", "rem", "zne", "pec")}
    ideal, shot_only = [], []
    for x in Xtest:
        z = vqc_predict(sim, x[None], theta, cfg.n_layers)[0]
        ideal.append(z)
        shot_only.append(np.mean(np.where(rng.random(cfg.shots) < (1 + z) / 2, 1.0, -1.0)))
        for m in rows:
            rows[m].append(qem_expectation(sim, x, theta, cfg, rng, m))
    ideal_arr = np.array(ideal)
    g = build_circuit(Xtest[0], theta, cfg.n_qubits, cfg.n_layers)
    gamma = run_noisy(sim, g, 1, cfg, rng, pec=True)[2]
    wz = richardson(cfg.zne_scales, [0, 0, 0])[1]
    out = []
    for name, est in [("Noiseless + shot noise", shot_only), ("Noisy (raw)", rows["raw"]),
                      ("REM only", rows["rem"]), ("ZNE (3-pt Richardson) + REM", rows["zne"]),
                      ("PEC + REM", rows["pec"])]:
        e = np.asarray(est) - ideal_arr
        out.append(dict(method=name, bias=float(e.mean()), RMSE=float(np.sqrt(np.mean(e ** 2))),
                        MAE=float(np.abs(e).mean())))
    df = pd.DataFrame(out)
    df["circuit_executions"] = [cfg.shots, cfg.shots, cfg.shots,
                                cfg.shots * len(cfg.zne_scales), cfg.shots]
    df["overhead_note"] = ["-", "-", "-", f"noise-amp. x{np.sqrt((wz ** 2).sum()):.2f}",
                           f"gamma_tot={gamma:.3f}"]
    return df


def confusion_matrix(n: int, cfg: Config) -> np.ndarray:
    m1 = np.array([[1 - cfg.ro_e01, cfg.ro_e10], [cfg.ro_e01, 1 - cfg.ro_e10]])
    M = np.array([[1.0]])
    for _ in range(n):
        M = np.kron(M, m1)
    return M


def rem_constrained_ls(M: np.ndarray, p_noisy: np.ndarray, w: float = 1e3) -> np.ndarray:
    """Eq. 12: Constrained least-squares PMF recovery (NNLS + sum-to-one constraint)."""
    A = np.vstack([M, w * np.ones((1, M.shape[1]))])
    b = np.r_[p_noisy, w]
    p, _ = nnls(A, b)
    return p / p.sum()


# ======================================================================================
# 5. ACTUARIAL RESERVING & SOLVENCY II CAPITAL METRICS (Eqs. 5, 13-16)
# ======================================================================================
def make_portfolio_sampler(mu_hat: np.ndarray, sigma: float, cfg: Config):
    """Aggregate portfolio loss sampler with systemic log-normal & Pareto jump contamination."""
    m_c = np.exp(mu_hat + sigma ** 2 / 2).sum()
    sd_c = np.sqrt((np.exp(sigma ** 2) - 1) * np.exp(2 * mu_hat + sigma ** 2).sum())

    def sample(n: int, rng: np.random.Generator) -> np.ndarray:
        sysf = np.exp(cfg.tau_sys * rng.standard_normal(n) - cfg.tau_sys ** 2 / 2)
        jump = (rng.random(n) < cfg.p_cat) * cfg.pareto_xm * (1 + rng.pareto(cfg.pareto_alpha, n))
        return np.maximum(sysf * (1 + jump) * (m_c + sd_c * rng.standard_normal(n)), 0)
    return sample


def discretise(samples: np.ndarray, n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    """Discretise aggregate losses into 2^n basis states (Eq. 5)."""
    K = 2 ** n_qubits
    edges = np.linspace(samples.min(), np.quantile(samples, 0.99995), K + 1)
    k = np.clip(np.digitize(samples, edges) - 1, 0, K - 1)
    cnt = np.bincount(k, minlength=K)
    sums = np.bincount(k, weights=samples, minlength=K)
    xk = np.where(cnt > 0, sums / np.maximum(cnt, 1), 0.5 * (edges[:-1] + edges[1:]))
    return xk, cnt / cnt.sum()


def risk_metrics(xk: np.ndarray, pk: np.ndarray, alpha: float) -> dict[str, float]:
    """Eqs. 13-16: BEL, VaR99.5%, TVaR99.5% (Rockafellar-Uryasev form), and SCR."""
    k = min(int(np.searchsorted(np.cumsum(pk), alpha)), len(pk) - 1)
    var = xk[k]
    bel = float(xk @ pk)
    tvar = var + float(np.maximum(xk - var, 0) @ pk) / (1 - alpha)
    return dict(BEL=bel, VaR=float(var), TVaR=float(tvar), SCR=float(var - bel))


# ======================================================================================
# 6. QUANTUM AMPLITUDE ESTIMATION (QAE) TAIL-RISK ENGINE (Eqs. 5-7)
# ======================================================================================
def grover_check(pk: np.ndarray, good: np.ndarray, m: int = 3) -> float:
    r"""Explicit verification of Grover operator Q = -A S_0 A^\dagger S_\chi (Eq. 6)."""
    amp = np.sqrt(pk)
    psi = np.r_[amp * np.sqrt(1 - good), amp * np.sqrt(good)]
    flags = np.r_[np.zeros(len(pk)), np.ones(len(pk))]
    s = psi.copy()
    for _ in range(m):
        s = s * (1 - 2 * flags)
        s = -(s - 2 * psi * (psi @ s))
    p = float(pk @ good)
    return abs(np.sqrt(np.sum(s[flags == 1] ** 2)) - abs(np.sin((2 * m + 1) * np.arcsin(np.sqrt(p)))))


def canonical_qae(p: float, m: int, rng: np.random.Generator, size: int = 1) -> np.ndarray:
    """QPE-based canonical QAE with M = 2^m evaluation points."""
    M, theta = 2 ** m, np.arcsin(np.sqrt(p)) / np.pi
    k = np.arange(M)
    pr = np.zeros(M)
    for w in (theta, 1 - theta):
        pr += 0.5 * np.abs(np.fft.fft(np.exp(2j * np.pi * k * w) / M)) ** 2
    pr /= pr.sum()
    y = rng.choice(M, size=size, p=pr)
    return np.sin(np.pi * y / M) ** 2


def mlae_schedule(budget: int, min_shots: int = 100, k_max: int = 10) -> tuple[list[int], int]:
    best = ([0], max(budget, 1))
    for K in range(1, k_max + 1):
        ms = [0] + [2 ** j for j in range(K)]
        C = sum(2 * m + 1 for m in ms)
        if budget // C < min_shots:
            break
        best = (ms, budget // C)
    return best


_TH = np.linspace(1e-5, np.pi / 2 - 1e-5, 20001)


def _fit_theta(cost, th=_TH):
    i = int(np.argmin(cost(th)))
    th2 = np.linspace(th[max(i - 2, 0)], th[min(i + 2, len(th) - 1)], 2001)
    return th2[int(np.argmin(cost(th2)))]


def mlae_binomial(ms: list, N: int, hits: np.ndarray) -> float:
    k = 2 * np.asarray(ms) + 1

    def cost(th):
        P = np.clip(np.sin(np.outer(th, k)) ** 2, 1e-12, 1 - 1e-12)
        return -(hits * np.log(P) + (N - hits) * np.log(1 - P)).sum(1)
    return float(np.sin(_fit_theta(cost)) ** 2)


def mlae_wls(ms: list, phat: np.ndarray, var: np.ndarray) -> float:
    k = 2 * np.asarray(ms) + 1

    def cost(th):
        return (((phat - np.sin(np.outer(th, k)) ** 2) ** 2) / var).sum(1)
    return float(np.sin(_fit_theta(cost)) ** 2)


def amplitude_estimate(p: float, budget: int, mode: str, cfg: Config,
                       rng: np.random.Generator) -> float:
    """MLAE tail probability estimator across noiseless and error-mitigated execution modes."""
    p = float(np.clip(p, 0.0, 1.0))
    if mode == "mc":
        return float(rng.binomial(budget, p) / budget)
    scales = cfg.zne_scales
    ms, N = mlae_schedule(budget // sum(scales) if mode == "noisy_zne" else budget)
    ms_arr = np.asarray(ms)
    P = np.sin((2 * ms_arr + 1) * np.arcsin(np.sqrt(p))) ** 2
    D = (1 - cfg.qae_eps) ** (2 * ms_arr + 1)
    if mode == "ideal":
        return mlae_binomial(ms, N, rng.binomial(N, P))
    Pn = 0.5 + (P - 0.5) * D
    if mode == "noisy_raw":
        return mlae_binomial(ms, N, rng.binomial(N, Pn))
    if mode == "noisy_zne":
        w = richardson(scales, [0] * len(scales))[1]
        est = np.array([rng.binomial(N, 0.5 + (P - 0.5) * D ** s) / N for s in scales])
        phat = w @ est

        def var_fn(P0):
            Pl = np.array([0.5 + (P0 - 0.5) * D ** s for s in scales])
            return ((w[:, None] ** 2) * Pl * (1 - Pl)).sum(0) / N + 1e-7
    elif mode == "noisy_pec":
        gam, wn = (2 - D) / D, 1 / (2 - D)
        n_noisy = rng.binomial(N, wn)
        g1, g2 = rng.binomial(n_noisy, Pn), rng.binomial(N - n_noisy, 0.5)
        phat = gam * (g1 - g2) / N

        def var_fn(P0):
            ev2 = gam ** 2 * (wn * (0.5 + (P0 - 0.5) * D) + (1 - wn) * 0.5)
            return np.maximum(ev2 - P0 ** 2, 1e-6) / N
    else:
        raise ValueError(mode)
    p0 = mlae_wls(ms, phat, np.ones_like(phat))
    P0 = np.sin((2 * ms_arr + 1) * np.arcsin(np.sqrt(p0))) ** 2
    return mlae_wls(ms, phat, var_fn(P0))


def qae_risk_metrics(xk: np.ndarray, pk: np.ndarray, cfg: Config, mode: str, budget: int,
                     rng: np.random.Generator) -> dict[str, float]:
    """Tail-risk evaluation via bisection on QAE tail estimates (Woerner-Egger)."""
    a = cfg.alpha
    if mode == "mc":
        s = xk[rng.choice(len(xk), budget, p=pk)]
        var = float(np.quantile(s, a))
        bel = float(s.mean())
        return dict(BEL=bel, VaR=var, TVaR=float(var + np.maximum(s - var, 0).mean() / (1 - a)),
                    SCR=float(var - bel))
    n_steps = int(np.ceil(np.log2(len(xk))))
    per_call = budget // (n_steps + 2)
    tail = np.r_[np.cumsum(pk[::-1])[::-1][1:], 0.0]
    lo, hi = 0, len(xk) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if amplitude_estimate(tail[mid], per_call, mode, cfg, rng) <= 1 - a:
            hi = mid
        else:
            lo = mid + 1
    var, x_top = xk[lo], xk.max()
    f = np.maximum(xk - var, 0) / (x_top - var + 1e-12)
    tvar = float(var + amplitude_estimate(pk @ f, per_call, mode, cfg, rng) * (x_top - var) / (1 - a))
    bel = float(amplitude_estimate(pk @ (xk / x_top), per_call, mode, cfg, rng) * x_top)
    return dict(BEL=bel, VaR=float(var), TVaR=tvar, SCR=float(var - bel))


def scaling_study(xk: np.ndarray, pk: np.ndarray, cfg: Config,
                  rng: np.random.Generator) -> tuple[pd.DataFrame, dict[str, float], float]:
    """RMSE scaling study comparing classical O(1/eps^2) Monte Carlo vs QAE O(1/eps)."""
    k = min(int(np.searchsorted(np.cumsum(pk), cfg.alpha)), len(pk) - 1)
    p_true = float(pk[k:].sum())
    rows = []
    for mode in ("mc", "ideal", "noisy_raw", "noisy_zne", "noisy_pec"):
        for b in cfg.scaling_budgets:
            est = np.array([amplitude_estimate(p_true, b, mode, cfg, rng)
                            for _ in range(cfg.scaling_reps)])
            err = est - p_true
            rows.append(dict(mode=mode, budget=b, rmse=float(np.sqrt(np.mean(err ** 2))),
                             median_abs_err=float(np.median(np.abs(err))), bias=float(err.mean()),
                             rel_rmse=float(np.sqrt(np.mean(err ** 2)) / p_true)))
    df = pd.DataFrame(rows)
    slopes = {m: float(np.polyfit(np.log(g.budget), np.log(g.median_abs_err + 1e-12), 1)[0])
              for m, g in df.groupby("mode")}
    return df, slopes, p_true


def canonical_bound_check(cfg: Config, rng: np.random.Generator, p: float = 0.005,
                          reps: int = 2000) -> pd.DataFrame:
    """Eq. 7: Canonical QAE analytical error bound check."""
    out = []
    for m in range(4, 11):
        M = 2 ** m
        est = canonical_qae(p, m, rng, reps)
        out.append(dict(m=m, M=M, bound=float(np.pi / M + np.pi ** 2 / M ** 2),
                        frac_within=float(np.mean(np.abs(est - p) <= np.pi / M + np.pi ** 2 / M ** 2)),
                        mean_abs_err=float(np.mean(np.abs(est - p)))))
    return pd.DataFrame(out)


def pmf_readout_study(xk: np.ndarray, pk: np.ndarray, cfg: Config,
                      rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Readout error simulation and constrained least-squares REM mitigation (Eq. 12)."""
    n = cfg.n_loss_qubits
    ks = np.repeat(np.arange(len(pk)), rng.multinomial(cfg.pmf_shots, pk))
    for b in range(n):
        bit = (ks >> b) & 1
        flip = rng.random(len(ks)) < np.where(bit == 0, cfg.ro_e01, cfg.ro_e10)
        ks = ks ^ (flip.astype(int) << b)
    p_noisy = np.bincount(ks, minlength=len(pk)) / len(ks)
    p_mit = rem_constrained_ls(confusion_matrix(n, cfg), p_noisy)
    rows = {"Exact pmf": pk, "Noisy readout": p_noisy, "REM (Eq. 12)": p_mit}
    return pd.DataFrame({k: risk_metrics(xk, v, cfg.alpha) for k, v in rows.items()}).T, p_mit


# ======================================================================================
# 7. MAIN PIPELINE EXECUTION
# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description="HQCAP Quantum-Classical Actuarial Framework")
    ap.add_argument("--synthetic", action="store_true", help="Run synthetic demonstration")
    ap.add_argument("--claims-csv", help="Path to input claims CSV")
    ap.add_argument("--target", help="Target column name")
    ap.add_argument("--features", help="Comma-separated list of feature columns")
    ap.add_argument("--out-dir", default="outputs", help="Output directory")
    ap.add_argument("--quick", action="store_true", help="Fast smoke test mode")
    args = ap.parse_args()

    cfg = Config(out_dir=args.out_dir)
    if args.quick:
        cfg.shots, cfg.spsa_iters, cfg.adam_iters = 2_000, 60, 60
        cfg.n_qem_test, cfg.scaling_reps, cfg.risk_reps = 6, 30, 6
        cfg.ref_samples, cfg.n_policyholders = 400_000, 10_000

    os.makedirs(cfg.out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    n, L = cfg.n_qubits, cfg.n_layers
    sim = QSim(n)

    # ---- Phase 1 Data Ingestion ----
    if args.claims_csv:
        cols = args.features.split(",")
        df = load_claims_csv(args.claims_csv, cols, args.target)
    else:
        df = make_claims(cfg.n_policyholders, rng)
    feats = [c for c in df.columns if c != "expenditure"][:n]
    X, logy = df[feats].to_numpy(float), np.log(df["expenditure"].to_numpy())

    perm = rng.permutation(len(df))
    itr, iva = perm[:cfg.n_train], perm[cfg.n_train:cfg.n_train + cfg.n_val]
    ite = perm[cfg.n_train + cfg.n_val:cfg.n_train + cfg.n_val + 2000]
    sc = Scaler().fit(X[itr], logy[itr])
    Xa, ys = sc.x(X), sc.y(logy)
    print(f"[HQCAP Data] Loaded {len(df)} policyholders, features = {feats}")

    # ---- Phase 1 VQC Model Optimization ----
    print("\n=== Phase 1: VQC Training & Parameter Optimization (Eqs. 1-4) ===")
    cands = {"SPSA": train_spsa(sim, Xa[itr], ys[itr], cfg, rng),
             "Adam+param-shift": train_adam_ps(sim, Xa[itr], ys[itr], cfg, rng)}
    res, best = [], None
    ridge = ridge_fit(Xa[itr], logy[itr])
    for name, (th, hist) in cands.items():
        f_tr = vqc_predict(sim, Xa[itr], th, L)
        a_, b_ = np.polyfit(f_tr, ys[itr], 1)
        pv = sc.inv_y(a_ * vqc_predict(sim, Xa[iva], th, L) + b_)
        pt = sc.inv_y(a_ * vqc_predict(sim, Xa[ite], th, L) + b_)
        res.append(dict(model=f"VQC-{name}", val_R2_log=r2(logy[iva], pv),
                        test_R2_log=r2(logy[ite], pt), test_RMSE_log=float(np.sqrt(np.mean((logy[ite] - pt) ** 2)))))
        if best is None or res[-1]["val_R2_log"] > best[0]:
            best = (res[-1]["val_R2_log"], name, th, a_, b_)
    pr = ridge(Xa[ite])
    res.append(dict(model="Ridge (classical)", val_R2_log=r2(logy[iva], ridge(Xa[iva])),
                    test_R2_log=r2(logy[ite], pr), test_RMSE_log=float(np.sqrt(np.mean((logy[ite] - pr) ** 2)))))
    vqc_tbl = pd.DataFrame(res)
    print(vqc_tbl.round(4).to_string(index=False))
    _, best_name, theta, a_cal, b_cal = best
    vqc_tbl.to_csv(f"{cfg.out_dir}/vqc_performance.csv", index=False)

    # ---- Phase 3 Error Mitigation Benchmark ----
    print(f"\n=== Phase 3: NISQ Error Mitigation Protocols (Best VQC: {best_name}) ===")
    qem = qem_benchmark(sim, Xa[ite[:cfg.n_qem_test]], theta, cfg, rng)
    print(qem.round(4).to_string(index=False))
    qem.to_csv(f"{cfg.out_dir}/qem_benchmark.csv", index=False)

    # ---- Phase 4 Portfolio Loss Distribution & Reserving ----
    print("\n=== Phase 4: Loss Register Discretization & Reserving Metrics ===")
    mu_hat = sc.inv_y(a_cal * vqc_predict(sim, Xa, theta, L) + b_cal)
    sigma = float(np.std(logy[ite] - sc.inv_y(a_cal * vqc_predict(sim, Xa[ite], theta, L) + b_cal)))
    sampler = make_portfolio_sampler(mu_hat, sigma, cfg)
    ref = sampler(cfg.ref_samples, rng)
    xk, pk = discretise(ref, cfg.n_loss_qubits)
    scale = float(xk @ pk)
    xk = xk / scale                                                 # Normalize in BEL units
    truth = risk_metrics(xk, pk, cfg.alpha)
    cont = dict(BEL=float(ref.mean() / scale), VaR=float(np.quantile(ref, cfg.alpha) / scale))
    print(f"Residual volatility sigma={sigma:.3f}; continuous-MC reference: BEL={cont['BEL']:.4f}, "
          f"VaR99.5={cont['VaR']:.4f}")
    pmf_tbl, p_mit = pmf_readout_study(xk, pk, cfg, rng)
    print(pmf_tbl.round(4).to_string())
    pmf_tbl.to_csv(f"{cfg.out_dir}/pmf_readout_rem.csv")

    # ---- Phase 2 QAE Acceleration & Tail Risk ----
    print("\n=== Phase 2: QAE Convergence Scaling & Tail Risk Engine ===")
    sc_df, slopes, p_true = scaling_study(xk, pk, cfg, rng)
    print(f"True tail probability p={p_true:.5f}; empirical log-log error slopes:")
    print({k: round(v, 3) for k, v in slopes.items()})
    print("RMSE Matrix:\n", sc_df.pivot(index="budget", columns="mode", values="rmse").round(5).to_string())
    sc_df.to_csv(f"{cfg.out_dir}/qae_scaling.csv", index=False)

    kk = min(int(np.searchsorted(np.cumsum(pk), cfg.alpha)), len(pk) - 1)
    print("Grover operator state check (Eq. 6) max deviation:",
          f"{grover_check(pk, (np.arange(len(pk)) >= kk).astype(float)):.2e}")
    cb = canonical_bound_check(cfg, rng)
    print("\nEq. 7 Canonical QAE analytical bound check:\n", cb.round(4).to_string(index=False))
    cb.to_csv(f"{cfg.out_dir}/qae_bound_check.csv", index=False)

    print(f"\n=== Solvency II Risk Metrics via QAE (Budget = {cfg.risk_budget} calls) ===")
    rows = [dict(method="Exact (discretised)", **truth, SCR_sd=0.0)]
    for mode in ("mc", "ideal", "noisy_raw", "noisy_zne", "noisy_pec"):
        r = pd.DataFrame([qae_risk_metrics(xk, pk, cfg, mode, cfg.risk_budget, rng)
                          for _ in range(cfg.risk_reps)])
        rows.append(dict(method=mode, **r.mean().to_dict(), SCR_sd=float(r["SCR"].std())))
    risk = pd.DataFrame(rows)
    print(risk.round(4).to_string(index=False))
    risk.to_csv(f"{cfg.out_dir}/risk_metrics_qae.csv", index=False)

    # ---- Visualizations ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for m, g in sc_df.groupby("mode"):
            ax[0].loglog(g.budget, g.median_abs_err, "o-", label=f"{m} (slope {slopes[m]:.2f})")
        ax[0].set_xlabel("Oracle calls")
        ax[0].set_ylabel("Median |error| of tail probability")
        ax[0].legend(fontsize=7)
        ax[1].semilogy(xk, pk, label="Exact PMF")
        ax[1].semilogy(xk, p_mit, label="REM Mitigated PMF", alpha=0.6)
        ax[1].axvline(truth["VaR"], ls="--", c="k", label="VaR 99.5%")
        ax[1].set_xlabel("Loss / BEL")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(f"{cfg.out_dir}/hqcap_figures.png", dpi=150)
        plt.close(fig)
        print(f"\nFigures saved successfully to {cfg.out_dir}/hqcap_figures.png")
    except Exception as exc:
        print("Plotting skipped:", exc)

    with open(f"{cfg.out_dir}/config.json", "w") as fh:
        json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()}, fh)

    print(f"\nHQCAP pipeline run finished successfully. All outputs written to '{cfg.out_dir}'.")


if __name__ == "__main__":
    main()
