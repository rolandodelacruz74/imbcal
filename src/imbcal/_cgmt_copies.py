"""Problema límite exacto (vía CGMT, derivación heurística) para copias contraídas: cada unidad minoritaria
aporta x_i y r − 1 copias de x̄₁ + √γ (x_i − x̄₁). Se separa z_i = z̄ + e_i (Σ e_i = 0): w = α μ/s + τ z̄/‖z̄‖ + θ u.
Ecuaciones (λ_r ≥ 0; F(e) = ℓ(A+e) + (r−1)ℓ(A+√γ e), A = αs + τ√(κ/π) + b, ‖w‖² = α² + τ² + θ²):
  e = prox_{λp F}(‖w‖Z − λp ν),  c = prox_{λp ℓ}(αs − b + ‖w‖Z₀)
  (b)  π E[F_A] = (1−π) E[ℓ'(c)]                     F_A = ℓ'(A+e) + (r−1)ℓ'(A+√γ e)
  (α)  α(Γ̄ + λ_r) = −s[π E F_A + (1−π) E ℓ'(c)]
  (τ)  τ(Γ̄ + λ_r) = −√(κπ) E F_A
  (θ)  Γ̄ + λ_r = κ/λp
  (ν)  E F'(e) = −ν                                      (media cero de la holgura minoritaria)
  (q)  λp² [π E(F'(e)+ν)² + (1−π) E ℓ'(c)²] = κ θ²
con Γ̄ = π E[F''/(1+λp F'')] + (1−π) E[ℓ''/(1+λp ℓ'')]. Con γ = 1 se recupera la logística con peso c = r."""
import sys, json, warnings, numpy as np
from scipy.optimize import root
from scipy.special import expit
from ._cgmt_weights import gh_x as gh_x_hermite, gh_w as gh_w_hermite
# Cuadratura: Gauss–Legendre con 1500 nodos en [−14, 14] contra la densidad normal. Cerca de la separabilidad
# (κ = 0,1) la norma de ŵ es grande y Gauss–Hermite con 160 nodos no basta (escala 6,242 en vez de 6,233).
from numpy.polynomial.legendre import leggauss as _lg
_x, _w = _lg(1500); gh_x = 14.0 * _x; gh_w = _w * 14.0 * np.exp(-gh_x**2 / 2); gh_w = gh_w / gh_w.sum()
l1 = lambda t: -expit(-t)                 # ℓ'
l2 = lambda t: expit(t) * expit(-t)       # ℓ''

def prox_F(v, lp, A, r, g):
    sg = np.sqrt(g); lo, hi = v - 1e-12, v + lp * (1 + (r - 1) * sg) + 1e-12
    for _ in range(80):
        e = 0.5 * (lo + hi); f = e + lp * (l1(A + e) + (r - 1) * sg * l1(A + sg * e)) - v
        lo = np.where(f < 0, e, lo); hi = np.where(f < 0, hi, e)
    return 0.5 * (lo + hi)

def prox_l(v, lp):
    lo, hi = v - 1e-12, v + lp + 1e-12
    for _ in range(80):
        c = 0.5 * (lo + hi); f = c + lp * l1(c) - v
        lo = np.where(f < 0, c, lo); hi = np.where(f < 0, hi, c)
    return 0.5 * (lo + hi)

def residuals(x, s, pi, kappa, r, g, lr):
    a, lt, lth, b, llp, nu = x
    tau, th, lp = lt, np.exp(lth), np.exp(llp)
    nw = np.sqrt(a * a + tau * tau + th * th); sg = np.sqrt(g)
    A = a * s + tau * np.sqrt(kappa / pi) + b
    e = prox_F(nw * gh_x - lp * nu, lp, A, r, g)
    FA = l1(A + e) + (r - 1) * l1(A + sg * e)
    Fp = l1(A + e) + (r - 1) * sg * l1(A + sg * e)
    Fpp = l2(A + e) + (r - 1) * g * l2(A + sg * e)
    c = prox_l(a * s - b + nw * gh_x, lp)
    EFA, ElpC = np.sum(gh_w * FA), np.sum(gh_w * l1(c))
    Gam = pi * np.sum(gh_w * Fpp / (1 + lp * Fpp)) + (1 - pi) * np.sum(gh_w * l2(c) / (1 + lp * l2(c)))
    EFp = np.sum(gh_w * Fp)
    q = lp**2 * (pi * np.sum(gh_w * (Fp + nu) ** 2) + (1 - pi) * np.sum(gh_w * l1(c) ** 2))
    return [pi * EFA - (1 - pi) * ElpC,
            a * (Gam + lr) + s * (pi * EFA + (1 - pi) * ElpC),
            tau * (Gam + lr) + np.sqrt(kappa * pi) * EFA,
            (Gam + lr) - kappa / lp,
            EFp + nu,
            (q - kappa * th**2) / max(kappa * th**2, 1e-12)]

def solve(s, pi, kappa, r, g, lr=0.0, x0=None):
    if x0 is None: x0 = np.array([2 * s, 0.1, np.log(0.3), -2.0, np.log(kappa / (pi * (1 - pi))), 0.0])
    best = None
    for method in ["hybr", "lm"]:
        rr = root(residuals, x0, args=(s, pi, kappa, r, g, lr), method=method, options={"xtol": 1e-13} if method == "hybr" else {"xtol": 1e-14, "ftol": 1e-14})
        res = float(np.max(np.abs(rr.fun)))
        if best is None or res < best[1]: best = (rr.x, res)
    x, res = best; a, tau, th = x[0], x[1], np.exp(x[2]); nw = np.sqrt(a * a + tau * tau + th * th)
    return dict(alpha=a, tau=tau, theta=th, b=x[3], lp=np.exp(x[4]), nu=x[5], scale=nw / (2 * s), cos=a / nw, res=res), x

def init_from_weights(s, pi, kappa, r):
    """Solución de pesos c = r (γ = 1) expresada en las variables de este sistema."""
    from . import _cgmt_weights as CP
    E = CP.path(s, pi, kappa, [r])[r]
    a, sig, lam, b = E["alpha"], E["sigma"], E["lam"], E["b"]
    nw = np.hypot(a, sig); t = CP.prox(a * s + b + nw * gh_x, lam * r); Epsi = np.sum(gh_w * r * expit(-t))
    tau = lam * Epsi * np.sqrt(pi / kappa); th = np.sqrt(max(sig**2 - tau**2, 1e-8))
    return np.array([a, tau, np.log(th), b, np.log(lam), Epsi])

def path(s, pi, kappa, r, g, lr=0.0, steps=20):
    """continuación en γ desde 1 (pesos c = r) hasta g, a κ fijo"""
    x = init_from_weights(s, pi, kappa, r)
    for gg in np.linspace(1.0, g, steps):
        rr = root(residuals, x, args=(s, pi, kappa, r, gg, lr), method="hybr", options={"xtol": 1e-13}); x = rr.x
    a, tau, th = x[0], x[1], np.exp(x[2]); nw = np.sqrt(a * a + tau * tau + th * th)
    return dict(alpha=a, tau=tau, theta=th, b=x[3], lp=np.exp(x[4]), nu=x[5], scale=nw / (2 * s), cos=a / nw,
                res=float(np.max(np.abs(rr.fun))))

