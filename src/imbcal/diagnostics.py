"""Diagnostics for synthetic minority data."""
from __future__ import annotations

import numpy as np


def majority_exposure(prob_fn, X_min, X_syn) -> dict:
    """Shift in estimated risk between synthetic and real minority points.

    ``delta = mean(prob_fn(X_syn)) - mean(prob_fn(X_min))``. With the true risk function this is the signed term of
    the weighted-bias identity ∫(1-η)(q-f1) = -delta: negative values mean that the generator moves minority mass
    towards regions dominated by the majority class, where the bias costs most. It is a descriptive heuristic: a
    positive value does not rule out local displacement that cancels on average.
    ``prob_fn`` maps an array of features to probabilities (e.g. a fitted model's ``predict_proba(X)[:, 1]``).
    """
    r = np.asarray(prob_fn(X_min), float); s = np.asarray(prob_fn(X_syn), float)
    return dict(delta=float(s.mean() - r.mean()), mean_real=float(r.mean()), mean_synthetic=float(s.mean()))
