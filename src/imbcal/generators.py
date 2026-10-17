"""Synthetic minority generators.

A generator draws ``m`` synthetic minority points from the observed minority sample. Any object with a
``sample(X_min, m, rng)`` method can be used, so deep generators (CTGAN, TVAE, diffusion models) plug in through a
thin wrapper.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class BaseGenerator:
    """Interface: ``sample(X_min, m, rng) -> ndarray of shape (m, n_features)``."""

    def sample(self, X_min: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({args})"


class SMOTEGenerator(BaseGenerator):
    """SMOTE interpolation towards one of the ``k`` nearest minority neighbours.

    ``k="sqrt"`` uses ``ceil(sqrt(n_min))`` neighbours, a wider kernel than the usual ``k=5``.
    """

    def __init__(self, k: int | str = 5):
        self.k = k

    def sample(self, X_min, m, rng):
        n1 = len(X_min)
        if n1 < 2:
            raise ValueError("SMOTE needs at least two minority points")
        k = int(np.ceil(np.sqrt(n1))) if self.k == "sqrt" else int(self.k)
        k = max(1, min(k, n1 - 1))
        nb = NearestNeighbors(n_neighbors=k + 1).fit(X_min).kneighbors(X_min, return_distance=False)[:, 1:]
        i = rng.integers(0, n1, m)
        j = nb[i, rng.integers(0, k, m)]
        return X_min[i] + rng.random((m, 1)) * (X_min[j] - X_min[i])


class GaussianGenerator(BaseGenerator):
    """Smoothed bootstrap: a resampled minority point plus ``N(0, h^2 I)`` noise.

    ``h`` is in the units of the features passed to ``fit`` (standardize first for a scale-free ``h``). With a
    Gaussian-kernel learner this is exactly a wider bandwidth for the minority class.
    """

    def __init__(self, h: float = 0.25):
        self.h = h

    def sample(self, X_min, m, rng):
        return X_min[rng.integers(0, len(X_min), m)] + self.h * rng.normal(size=(m, X_min.shape[1]))


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
