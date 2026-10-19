"""Internal solver (ported from the replication package of de la Cruz, Paper A).

Sistema de punto fijo (CGMT / leave-one-out) para la logística SIN regularizar con pesos de clase
en el 2-GMM, régimen proporcional d/n → κ, por debajo del umbral de separabilidad.

Modelo: y = ±1 con P(y = 1) = π; x = yμ + z, z ~ N(0, I_d), ‖μ‖ = s.
Estimador: min_{w,b} Σ_i c(y_i) ℓ(y_i(⟨w, x_i⟩ + b)), ℓ(t) = log(1 + e^{−t}), c(+1) = c, c(−1) = 1.
Descomposición: w = α μ/s + σ u, u ⊥ μ unitario. Margen de entrenamiento (con corrección LOO):
    t = prox_{λ c(y) ℓ}(v),   v = α(s + H) + σ G + y b,   H, G ~ N(0, 1) independientes,
    ψ = c(y) σ(−t) = (t − v)/λ ≥ 0.
Ecuaciones (incógnitas α, σ, λ, b):
    (E1) E[y ψ] = 0                        (intercepto)
    (E2) E[ψ (s + H)] = 0                  (estacionariedad en la dirección de μ)
    (E3) λ² E[ψ²] = κ σ²                   (norma de la parte ortogonal)
    (E4) E[1 / (1 + λ c(y) ℓ''(t))] = 1 − κ  (grados de libertad)
Predicción para x nuevo: logit η̂ = α(y s + h) + σ g + b − log c. Verdad: 2(y s² + s h) + logit π.
Con c = 1/π·(1−π) se obtiene "pesos balanceados". κ → 0 recupera α = 2s, σ = 0, b = logit π + log c.
"""
import numpy as np
from scipy.optimize import fsolve
from scipy.special import expit, logit

NODES = 160
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(NODES)
gh_w = gh_w / gh_w.sum()
# Reducción a una dimensión: v = α s + y b + τ Z con τ² = α² + σ², y por Stein
# E[ψ H] = α E[dψ/dv], dψ/dv = −c ℓ''(t) / (1 + λ c ℓ''(t)).

def prox(v, lam_c, it=8):
    """t = v + λc σ(−t): raíz única en [v, v + λc]; bisección y luego Newton."""
    lo, hi = v.copy(), v + lam_c
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        f = mid - v - lam_c * expit(-mid)
        lo = np.where(f < 0, mid, lo); hi = np.where(f < 0, hi, mid)
    t = 0.5 * (lo + hi)
    for _ in range(it):
        t = t - (t - v - lam_c * expit(-t)) / (1 + lam_c * expit(t) * expit(-t))
    return t

def residuals(par, s, pi, kappa, c):
    a, ls, ll, b = par
    sig, lam = np.exp(ls), np.exp(ll)
    tau = np.hypot(a, sig)
    e1 = e2 = e3 = e4 = 0.0
    for y, py, cy in [(1, pi, c), (-1, 1 - pi, 1.0)]:
        v = a * s + y * b + tau * gh_x
        t = prox(v, lam * cy)
        psi = cy * expit(-t)
        l2 = cy * expit(t) * expit(-t)
        dpsi = -l2 / (1 + lam * l2)
        e1 += py * np.sum(gh_w * y * psi)
        e2 += py * np.sum(gh_w * (s * psi + a * dpsi))
        e3 += py * np.sum(gh_w * psi**2)
        e4 += py * np.sum(gh_w / (1 + lam * l2))
    return [e1, e2, lam**2 * e3 - kappa * sig**2, e4 - (1 - kappa)]

def solve(s, pi, kappa, c, x0=None):
    from scipy.optimize import root
    if x0 is None:
        lam = kappa / (pi * (1 - pi))
        x0 = [2 * s, np.log(np.sqrt(20 * kappa)), np.log(lam), logit(pi) + np.log(c)]
    r = root(residuals, x0, args=(s, pi, kappa, c), method="hybr", options={"xtol": 1e-12})
    a, ls, ll, b = r.x
    return dict(alpha=a, sigma=np.exp(ls), lam=np.exp(ll), b=b, ok=bool(r.success),
                res=float(np.max(np.abs(r.fun)))), r.x

def metrics(sol, s, pi, c, nmc=400000, seed=0):
    a, sig, b = sol["alpha"], sol["sigma"], sol["b"]
    rng = np.random.default_rng(seed)
    y = np.where(rng.random(nmc) < pi, 1, -1); h = rng.normal(size=nmc); g = rng.normal(size=nmc)
    zt = 2 * (y * s**2 + s * h) + logit(pi)
    zh = a * (y * s + h) + sig * g + b - np.log(c)
    eta, p = expit(zt), expit(zh)
    yy = (y == 1).astype(float)
    from sklearn.linear_model import LogisticRegression
    th = [0, LogisticRegression(C=np.inf, max_iter=5000).fit(zh.reshape(-1, 1), yy).coef_[0, 0]]
    roi = eta >= 0.05
    err = np.abs(p - eta)
    nb = 2 * s; nrm = np.hypot(a, sig)
    return dict(coseno=a / nrm, norma_rel=nrm / nb, proy=a / nb, ortog=sig / nb, slope=th[1],
                mae_eta=err.mean(), mae_eta_roi=err[roi].mean(),
                logloss=-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))

def _solve_robust(s, pi, kappa, c, x0):
    from scipy.optimize import root
    sol, x = solve(s, pi, kappa, c, x0)
    if not sol["ok"] or sol["res"] > 1e-9:
        r = root(residuals, x0, args=(s, pi, kappa, c), method="lm", options={"xtol": 1e-14, "ftol": 1e-14})
        a, ls, ll, b = r.x
        res = float(np.max(np.abs(r.fun)))
        if res < sol["res"]:
            sol, x = dict(alpha=a, sigma=np.exp(ls), lam=np.exp(ll), b=b, ok=res < 1e-9, res=res), r.x
    return sol, x

def path(s, pi, kappa, cs, steps_per_unit=30):
    """Continuación fina: en κ desde 0.005 con c = 1, luego en log c hacia arriba y hacia abajo."""
    x0 = None
    for k in np.linspace(min(0.005, kappa), kappa, 60):
        _, x0 = _solve_robust(s, pi, k, 1.0, x0) if x0 is not None else solve(s, pi, k, 1.0, None)
    out = {1.0: solve(s, pi, kappa, 1.0, x0)[0]}
    for direction in (+1, -1):
        targets = sorted([c for c in cs if (c > 1 if direction > 0 else c < 1)], key=lambda c: abs(np.log(c)))
        if not targets: continue
        lmax = abs(np.log(targets[-1]))
        grid = np.linspace(0, lmax, max(2, int(lmax * steps_per_unit)))
        grid = np.unique(np.r_[grid, [abs(np.log(c)) for c in targets]])
        x = x0.copy()
        for lg in grid[1:]:
            c = float(np.exp(direction * lg))
            sol, x = _solve_robust(s, pi, kappa, c, x)
            for ct in targets:
                if abs(abs(np.log(ct)) - lg) < 1e-12:
                    out[ct] = sol
    return out

