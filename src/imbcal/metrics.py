"""Metrics for probability estimation and paired comparisons over repeated splits."""
from __future__ import annotations

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def calibration_slope_intercept(y, p, eps: float = 1e-6):
    """Logistic recalibration of ``logit(p)``: returns (slope, intercept). Ideal values: 1 and 0."""
    z = logit(np.clip(np.asarray(p, float), eps, 1 - eps)).reshape(-1, 1)
    m = LogisticRegression(C=1e6).fit(z, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def expected_calibration_error(y, p, n_bins: int = 10, strategy: str = "quantile"):
    """ECE with equal-frequency (default) or equal-width bins. A secondary, binning-dependent summary."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(y[sel].mean() - p[sel].mean())
    return float(ece)


def probability_report(y, p) -> dict:
    """Log-loss, Brier score, AUC, calibration slope and intercept, ECE and mean predicted probability."""
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    slope, intercept = calibration_slope_intercept(y, p)
    return dict(log_loss=float(log_loss(y, p, labels=[0, 1])), brier=float(brier_score_loss(y, p)),
                auc=float(roc_auc_score(y, p)) if 0 < np.sum(y) < len(y) else float("nan"),
                calibration_slope=slope, calibration_intercept=intercept,
                ece=expected_calibration_error(y, p), mean_prediction=float(p.mean()), prevalence=float(np.mean(y)))


def paired_comparison(scores_a, scores_b, n_train: int, n_test: int) -> dict:
    """Compare two methods evaluated on the same B repeated splits (lower score = better).

    Returns the mean difference ``a - b``, the naive and the Nadeau–Bengio corrected standard errors,
    sd(D) * sqrt(1/B + n_test/n_train), and the corrected z statistic.
    """
    d = np.asarray(scores_a, float) - np.asarray(scores_b, float)
    B = len(d)
    if B < 2:
        raise ValueError("need at least two splits")
    sd = d.std(ddof=1)
    se_naive = sd / np.sqrt(B)
    se_nb = sd * np.sqrt(1 / B + n_test / n_train)
    return dict(mean_difference=float(d.mean()), se_naive=float(se_naive), se_corrected=float(se_nb),
                z_corrected=float(d.mean() / se_nb) if se_nb > 0 else 0.0, n_splits=B)
