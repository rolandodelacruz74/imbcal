"""Synthetic minority generators.

A generator draws ``m`` synthetic minority points from the observed minority sample. Any object with a
``sample(X_min, m, rng)`` method can be used, so deep generators (CTGAN, TVAE, diffusion models) plug in through a
thin wrapper.

The built-in :class:`SMOTEGenerator` and :class:`GaussianGenerator` perturb features continuously. With binary or
one-hot encoded columns they produce fractional values there; pass ``continuous_features`` to perturb only the
continuous columns and copy the others from the parent point.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _feature_mask(continuous_features, n_features):
    """Boolean mask of the perturbed columns from ``None`` (all), integer indices or a boolean mask."""
    if continuous_features is None:
        return np.ones(n_features, bool)
    cf = np.asarray(continuous_features)
    if cf.dtype == bool:
        if cf.shape != (n_features,):
            raise ValueError(f"continuous_features mask has shape {cf.shape}, expected {(n_features,)}")
        return cf.copy()
    if cf.ndim != 1 or not np.issubdtype(cf.dtype, np.integer):
        raise ValueError("continuous_features must be None, a list of column indices or a boolean mask")
    if cf.size and (cf.min() < -n_features or cf.max() >= n_features):
        raise ValueError(f"continuous_features index out of range for {n_features} features")
    mask = np.zeros(n_features, bool)
    mask[cf] = True
    return mask


class BaseGenerator:
    """Interface: ``sample(X_min, m, rng) -> ndarray of shape (m, n_features)``.

    Subclasses that move points continuously in feature space set ``perturbs_continuously = True`` and implement
    ``perturbed_features(n_features)``; the estimator then warns when a perturbed column is binary.
    """

    perturbs_continuously = False

    def sample(self, X_min: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def perturbed_features(self, n_features: int) -> np.ndarray:
        """Boolean mask of the columns this generator perturbs."""
        return _feature_mask(getattr(self, "continuous_features", None), n_features)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items()
                         if not (k == "continuous_features" and v is None))
        return f"{type(self).__name__}({args})"


class SMOTEGenerator(BaseGenerator):
    """SMOTE interpolation towards one of the ``k`` nearest minority neighbours.

    ``k="sqrt"`` uses ``ceil(sqrt(n_min))`` neighbours, a wider kernel than the usual ``k=5``.
    ``continuous_features`` (column indices or a boolean mask; default all columns): only these columns are
    interpolated, the others are copied from the parent point. Neighbours are found on all columns.
    """

    perturbs_continuously = True

    def __init__(self, k: int | str = 5, continuous_features=None):
        self.k = k
        self.continuous_features = continuous_features

    def sample(self, X_min, m, rng):
        n1 = len(X_min)
        if n1 < 2:
            raise ValueError("SMOTE needs at least two minority points")
        k = int(np.ceil(np.sqrt(n1))) if self.k == "sqrt" else int(self.k)
        k = max(1, min(k, n1 - 1))
        nb = NearestNeighbors(n_neighbors=k + 1).fit(X_min).kneighbors(X_min, return_distance=False)[:, 1:]
        i = rng.integers(0, n1, m)
        j = nb[i, rng.integers(0, k, m)]
        step = rng.random((m, 1)) * (X_min[j] - X_min[i])
        if self.continuous_features is None:
            return X_min[i] + step
        mask = self.perturbed_features(X_min.shape[1])
        out = np.array(X_min[i], dtype=float)
        out[:, mask] += step[:, mask]
        return out


class GaussianGenerator(BaseGenerator):
    """Smoothed bootstrap: a resampled minority point plus ``N(0, h^2 I)`` noise.

    ``h`` is in the units of the features passed to ``fit`` (standardize first for a scale-free ``h``). With a
    Gaussian-kernel learner this is exactly a wider bandwidth for the minority class.
    ``continuous_features`` (column indices or a boolean mask; default all columns): noise is added only to these
    columns, the others are copied from the resampled parent point.
    """

    perturbs_continuously = True

    def __init__(self, h: float = 0.25, continuous_features=None):
        self.h = h
        self.continuous_features = continuous_features

    def sample(self, X_min, m, rng):
        parents = X_min[rng.integers(0, len(X_min), m)]
        if self.continuous_features is None:
            return parents + self.h * rng.normal(size=(m, X_min.shape[1]))
        mask = self.perturbed_features(X_min.shape[1])
        out = np.array(parents, dtype=float)
        out[:, mask] += self.h * rng.normal(size=(m, int(mask.sum())))
        return out


class CallableGenerator(BaseGenerator):
    """Wrap a function ``f(X_min, m, rng) -> ndarray``."""

    def __init__(self, func, name: str = "callable"):
        self.func = func
        self.name = name

    def sample(self, X_min, m, rng):
        out = np.asarray(self.func(X_min, m, rng), dtype=float)
        if out.shape != (m, X_min.shape[1]):
            raise ValueError(f"generator returned shape {out.shape}, expected {(m, X_min.shape[1])}")
        return out

    def __repr__(self) -> str:
        return f"CallableGenerator({self.name})"
