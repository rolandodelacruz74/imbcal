"""Diagnostics for synthetic minority data."""
from __future__ import annotations

import numpy as np


def majority_exposure(prob_fn, X_min, X_syn) -> dict:
    """Shift in estimated risk between synthetic and real minority points (heuristic diagnostic).

    ``delta = mean(prob_fn(X_syn)) - mean(prob_fn(X_min))``, where ``prob_fn`` maps an array of features to
    probabilities (e.g. a fitted model's ``predict_proba(X)[:, 1]``).

    This is the plug-in analogue of the population risk shift ``Delta eta`` of the companion paper: with the true risk
    function ``eta`` and the generator law ``q`` in place of the synthetic sample, ``-Delta eta`` is the signed term
    ``integral (1 - eta)(q - f1)`` of the weighted-bias identity. Negative values suggest that the generator moves
    minority mass towards regions dominated by the majority class, where the bias costs most.

    It is a heuristic, not a test. Even the population quantity is only a moderate predictor of harm: across 20
    simulated schemes its rank correlation with the harm after recalibration was about -0.49. The plug-in version, which
    uses an estimated risk function and a finite synthetic sample, is not covered by any theory. A value near zero or
    positive does not rule out local displacements that cancel on average.
    """
    r = np.asarray(prob_fn(X_min), float); s = np.asarray(prob_fn(X_syn), float)
    return dict(delta=float(s.mean() - r.mean()), mean_real=float(r.mean()), mean_synthetic=float(s.mean()))
