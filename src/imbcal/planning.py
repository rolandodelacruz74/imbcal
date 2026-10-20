"""rebalancing_cost: what will rebalancing cost before you fit the model?

Asymptotic prediction (proportional regime, two-class Gaussian mixture; Theorems 2 and 5 of the paper, the latter
solved at zero penalty) of the validation calibration slope, AUC and log-loss of an unpenalized logistic regression fitted
  - by maximum likelihood ("mle"),
  - with balanced class weights followed by the intercept correction ("weights"),
  - after random undersampling to balance, with the intercept correction ("undersampling"),
  - after balanced SMOTE, with the intercept correction ("smote"; contracted-copies model),
together with the van Houwelingen shrinkage factor used in sample-size criteria (Riley et al., 2019).

Inputs are quantities that can be specified before data collection: the number of model parameters p, the
outcome prevalence, either the development sample size n or the events per variable (EPV), and the anticipated
AUC of a well-fitted model.

What it is: a closed-form planning screen under a Gaussian model (two-class Gaussian mixture), not a sample-size
calculation for the data at hand. Its maximum-likelihood slope predictions agree closely with simulation-based
sample-size tools that use only (p, prevalence, anticipated AUC), in the style of the R package samplesizedev of
Pavlou et al. It is optimistic relative to real data with many dummy (binary) predictors: in the two applications of the
paper the observed damage was at least as large, more so at low EPV. Read it asymmetrically: a predicted calibration
slope below 0.9 indicates a design that is too small, while a predicted slope above 0.9 does not guarantee adequacy.
For sample-size planning of prediction models see Riley et al. (2019, Statistics in Medicine 38:1276-1296) and
Riley et al. (2025, arXiv:2504.18730).

Usage:
  >>> from imbcal.planning import rebalancing_cost, print_rebalancing_cost
  >>> print_rebalancing_cost(rebalancing_cost(p=71, prevalence=0.092, epv=2.5, auc=0.81))
"""
import warnings
import numpy as np
from scipy.stats import norm
from scipy.special import expit, logit
from scipy.optimize import brentq
from . import _cgmt_weights as CP
from . import _cgmt_copies as CC

GH_X, GH_W = CP.gh_x, CP.gh_w

def _weights_solution(s, pi, kappa, c):
    """Unpenalized class-weighted logistic regression (Theorem 2): returns (m, q, b) with continuation in kappa and c."""
    sol = CP.path(s, pi, kappa, [c])
    S = sol[c] if c != 1 else sol[1.0]
    a, sig = S["alpha"], S["sigma"]
    return a * s, a * a + sig * sig, S["b"], bool(S["ok"])

def _evaluate(m, q, b_eff, s, pi):
    """Score T | y ~ N(y m, q); predicted log-odds T + b_eff; returns slope, AUC, log-loss (exact under the mixture)."""
    rho = m / (s * np.sqrt(q))
    ll = 0.0
    for y, py in [(1, pi), (-1, 1 - pi)]:
        z = y * m + np.sqrt(q) * GH_X + b_eff
        ll += py * np.sum(GH_W * (np.logaddexp(0, -z) if y == 1 else np.logaddexp(0, z)))
    return dict(calibration_slope=2 * m / q, auc=float(norm.cdf(np.sqrt(2) * s * rho)), log_loss=float(ll))

def rebalancing_cost(p, prevalence, auc, n=None, epv=None, smote_contraction=2 / 3):
    """Predicted validation metrics for each method (closed-form Gaussian-model planning screen).

    Give n (development sample size) or epv (events per parameter).
    The predictions are optimistic for real data with many dummy predictors: a predicted calibration slope below 0.9
    flags a design that is too small, but a slope above 0.9 does not guarantee adequacy (see the module docstring and
    Riley et al., 2019, 2025).
    auc: anticipated AUC of a well-fitted model, used to set the separation s = Phi^{-1}(auc)/sqrt(2).
    smote_contraction: squared-distance contraction of SMOTE points toward the minority mean (2/3 in high dimension,
    Lemma 2; larger, e.g. 0.8-0.9, with many binary predictors)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _rebalancing_cost(p, prevalence, auc, n=n, epv=epv, smote_contraction=smote_contraction)


def _rebalancing_cost(p, prevalence, auc, n=None, epv=None, smote_contraction=2 / 3):
    if (n is None) == (epv is None):
        raise ValueError("give exactly one of n or epv")
    pi = float(prevalence)
    n = float(n) if n is not None else epv * p / pi
    kappa = p / n
    s = norm.ppf(auc) / np.sqrt(2)
    r = (1 - pi) / pi
    out = {}
    m, q, b, ok = _weights_solution(s, pi, kappa, 1.0); out["mle"] = _evaluate(m, q, b, s, pi)
    m, q, b, ok = _weights_solution(s, pi, kappa, r); out["weights"] = _evaluate(m, q, b - np.log(r), s, pi)
    m, q, b, ok = _weights_solution(s, 0.5, kappa / (2 * pi), 1.0); out["undersampling"] = _evaluate(m, q, b - np.log(r), s, pi)
    sol = CC.path(s, pi, kappa, r, smote_contraction)
    nw2 = sol["alpha"] ** 2 + sol["tau"] ** 2 + sol["theta"] ** 2
    A = sol["alpha"] * s                                           # m = <w, mu>
    out["smote"] = _evaluate(A, nw2, sol["b"] - np.log(r), s, pi)
    # van Houwelingen shrinkage factor with R2_CS of the Bayes model
    H = -(pi * np.log(pi) + (1 - pi) * np.log(1 - pi))
    Lb = _evaluate(2 * s * s, 4 * s * s, logit(pi), s, pi)["log_loss"]
    R2 = 1 - np.exp(-2 * (H - Lb))
    vh = brentq(lambda S: (S - 1) * np.log(1 - R2 / S) - kappa, R2 + 1e-9, 1 - 1e-12)
    return dict(inputs=dict(p=p, prevalence=pi, n=n, epv=pi * n / p, kappa=kappa, auc=auc, separation=s,
                            smote_contraction=smote_contraction),
                methods=out, van_houwelingen_shrinkage=vh)

def print_rebalancing_cost(res):
    i = res["inputs"]
    print(f"p={i['p']}  prevalence={i['prevalence']:.3f}  n={i['n']:.0f}  EPV={i['epv']:.2f}  kappa=p/n={i['kappa']:.4f}  "
          f"anticipated AUC={i['auc']:.3f}")
    print(f"{'method':<15}{'cal. slope':>11}{'AUC':>8}{'log-loss':>10}")
    for k, v in res["methods"].items():
        print(f"{k:<15}{v['calibration_slope']:>11.3f}{v['auc']:>8.3f}{v['log_loss']:>10.4f}")
    print(f"van Houwelingen shrinkage factor (maximum likelihood): {res['van_houwelingen_shrinkage']:.3f}")
    print("Closed-form Gaussian-model screen, optimistic with many dummy predictors: slope < 0.9 flags a design that is "
          "too small; slope >= 0.9 does not guarantee adequacy.")

