"""Recalibration maps fitted on out-of-fold scores."""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-4


def to_logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    return logit(np.clip(np.asarray(p, float), eps, 1 - eps))


class PlattCalibrator:
    """Logistic recalibration ``expit(a + b * logit(p))`` (Platt scaling on the logit scale)."""

    def fit(self, p, y):
        self.model_ = LogisticRegression(C=1e6).fit(to_logit(p).reshape(-1, 1), y)
        self.intercept_ = float(self.model_.intercept_[0])
        self.slope_ = float(self.model_.coef_[0, 0])
        return self

    def predict(self, p):
        return expit(self.intercept_ + self.slope_ * to_logit(p))


class IsotonicCalibrator:
    """Isotonic recalibration (needs many events; unstable with fewer than a few hundred)."""

    def fit(self, p, y):
        self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(p, y)
        return self

    def predict(self, p):
        return self.model_.predict(p)


class IdentityCalibrator:
    def fit(self, p, y):
        return self

    def predict(self, p):
        return np.asarray(p, float)


def make_calibrator(name):
    if name in (None, "none"):
        return IdentityCalibrator()
    if name == "platt":
        return PlattCalibrator()
    if name == "isotonic":
        return IsotonicCalibrator()
    raise ValueError(f"unknown calibration {name!r}")


def crossfit_calibrated(p_oof: np.ndarray, y: np.ndarray, folds, calibration: str) -> np.ndarray:
    """Recalibrate out-of-fold scores without optimism: the map for fold j is fitted on the other folds."""
    out = np.empty(len(y))
    for tr, va in folds:
        out[va] = make_calibrator(calibration).fit(p_oof[tr], y[tr]).predict(p_oof[va])
    return out
