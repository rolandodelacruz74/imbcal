"""Probability estimation under class imbalance with a scikit-learn compatible estimator."""
from __future__ import annotations

from dataclasses import dataclass

import itertools

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

from .calibration import crossfit_calibrated, make_calibrator
from .generators import GaussianGenerator, SMOTEGenerator


@dataclass(frozen=True)
class Candidate:
    """A training scheme: no correction (generator None, chi 1) or prevalence-preserving augmentation."""
    generator: object
    chi: float

    def label(self) -> str:
        if self.generator is None:
            return "none"
        if self.generator == "weights":
            return "weights"
        return f"{self.generator!r}|chi={self.chi:g}"


def _loss(name, y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    if name == "log_loss":
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    if name == "brier":
        return (p - y) ** 2
    raise ValueError(f"unknown criterion {name!r}")


class ImbalanceCalibratedClassifier(ClassifierMixin, BaseEstimator):
    """Wrap any probabilistic classifier for calibrated probabilities at the natural prevalence.

    Parameters
    ----------
    estimator : scikit-learn classifier with ``sample_weight`` support in ``fit`` and ``predict_proba``.
    strategy : {"none", "weights", "augment"}
        ``"none"``: fit at the natural prevalence (recommended default).
        ``"weights"``: balanced class weights with recalibration (provided for comparison; it usually hurts).
        ``"augment"``: prevalence-preserving augmentation. Real minority points get weight ``chi`` and ``m``
        synthetic points from a generator get weight ``(1 - chi) n_min / m``, so the training prevalence is the
        natural one. ``(generator, chi)`` is chosen by cross-validation of the *recalibrated* loss, with
        ``chi = 1`` (no augmentation) always among the candidates.
    generators : list of generators for ``"augment"``. Default: Gaussian smoothed bootstrap with h in {0.25, 0.5}
        and SMOTE with k = 5 and k = sqrt(n_min). Features should be on a common scale.
    chi_grid : weights of the real minority points for ``"augment"`` (1 is always added).
    calibration : {"platt", "isotonic", None}. Fitted on out-of-fold scores of the selected scheme.
    criterion : {"log_loss", "brier"} used for selection.
    selection : {"best", "1se", "safe"}. ``"1se"`` keeps ``chi = 1`` unless the best candidate improves on it by more
        than one standard error of the per-observation loss difference. ``"safe"`` augments only if some candidate
        improves on ``chi = 1`` with simultaneous significance at level ``alpha``: paired cross-validated loss
        differences, within-fold variance (Bayle, Bayle, Janson and Mackey, 2020) and a max-T critical value from the
        estimated correlation between candidates; among the candidates that pass, the one with the smallest criterion.
        Asymptotically, the probability of selecting a candidate that is not better than ``chi = 1`` is at most
        ``alpha``.
    alpha : level of the ``"safe"`` selection (accepted risk of selecting a candidate that does not improve).
    variance_floor : for ``"safe"``, each candidate's standard deviation is replaced by
        ``max(sd, variance_floor * sd(loss of chi = 1))``; this makes the guarantee hold without fixing in advance which
        candidates are non-degenerate (a floor can only make the test more conservative).
    n_synthetic : number of synthetic points; default ``n_majority - n_minority``.
    cv : number of stratified folds.
    nested : {"aligned", "crossfit"}
        How the recalibration enters the selection criterion. ``"aligned"`` (default) is strictly nested: the
        candidate evaluated on fold v, including its recalibration map, is built from the other folds only; the map of
        fold v is fitted on scores of fold u != v from models trained without folds u and v, so each candidate costs
        ``cv + cv(cv-1)/2`` fits. ``"crossfit"`` (v0.1 behaviour) fits the map of fold v on the out-of-fold scores of
        the other folds, whose models saw fold v; cheaper (``cv`` fits) but not covered by the oracle inequality.
    output : {"ensemble", "refit"}
        ``"ensemble"`` (default) predicts with the average of the ``cv`` recalibrated fold predictors of the selected
        candidate, which inherits the guarantee of the selection by convexity of the loss and needs no extra fit.
        ``"refit"`` refits the selected candidate on all data and recalibrates it with the map fitted on the
        out-of-fold scores.
    stratify : bool
        Stratified folds (default, more stable with few events). The oracle inequality assumes folds independent of
        the data, i.e. ``stratify=False``.
    random_state : int or None.

    Attributes
    ----------
    selected_ : Candidate chosen.
    cv_results_ : dict label -> cross-validated recalibrated loss.
    candidate_distance_ : root mean squared difference between the out-of-fold probabilities of the selected
        scheme and those of ``chi = 1`` (controls the worst-case loss of the selection, Prop. 4 of the paper).
    calibrator_ : fitted recalibration map (``output="refit"``).
    estimators_, calibrators_ : fold predictors and their maps (``output="ensemble"``).
    """

    def __init__(self, estimator, strategy="none", generators=None, chi_grid=(0.0, 0.1, 0.25, 0.5, 0.75),
                 calibration="platt", criterion="log_loss", selection="best", n_synthetic=None, cv=5,
                 nested="aligned", output="ensemble", stratify=True, alpha=0.2, variance_floor=0.05,
                 random_state=None):
        self.estimator = estimator
        self.strategy = strategy
        self.generators = generators
        self.chi_grid = chi_grid
        self.calibration = calibration
        self.criterion = criterion
        self.selection = selection
        self.n_synthetic = n_synthetic
        self.cv = cv
        self.nested = nested
        self.output = output
        self.stratify = stratify
        self.alpha = alpha
        self.variance_floor = variance_floor
        self.random_state = random_state

    # ------------------------------------------------------------------ internals
    def _candidates(self):
        if self.strategy == "none":
            return [Candidate(None, 1.0)]
        if self.strategy == "weights":
            return [Candidate("weights", 1.0)]
        if self.strategy != "augment":
            raise ValueError(f"unknown strategy {self.strategy!r}")
        gens = self.generators if self.generators is not None else [
            GaussianGenerator(0.25), GaussianGenerator(0.5), SMOTEGenerator(5), SMOTEGenerator("sqrt")]
        return [Candidate(None, 1.0)] + [Candidate(g, float(c)) for g in gens for c in self.chi_grid if c < 1]

    def _fit_predict(self, X, y, Xt, cand, rng, synth_cache=None):
        est = clone(self.estimator)
        if cand.generator is None:
            est.fit(X, y)
        elif cand.generator == "weights":
            n1 = y.sum(); r = (len(y) - n1) / n1
            est.fit(X, y, sample_weight=np.where(y == 1, r, 1.0))
        else:
            n1 = int(y.sum()); n0 = len(y) - n1
            m = int(self.n_synthetic) if self.n_synthetic else max(n0 - n1, n1)
            key = id(cand.generator)
            if synth_cache is not None and key in synth_cache:
                S = synth_cache[key]
            else:
                S = cand.generator.sample(X[y == 1], m, rng)
                if synth_cache is not None:
                    synth_cache[key] = S
            XX = np.vstack([X, S]); YY = np.r_[y, np.ones(len(S), int)]
            W = np.r_[np.where(y == 1, cand.chi, 1.0), np.full(len(S), (1 - cand.chi) * n1 / len(S))]
            est.fit(XX, YY, sample_weight=W)
        return est, np.clip(est.predict_proba(Xt)[:, 1], 1e-6, 1 - 1e-6)

    # ------------------------------------------------------------------ API
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("binary classification only")
        if self.nested not in ("aligned", "crossfit"):
            raise ValueError(f"unknown nested {self.nested!r}")
        if self.output not in ("ensemble", "refit"):
            raise ValueError(f"unknown output {self.output!r}")
        if self.selection not in ("best", "1se", "safe"):
            raise ValueError(f"unknown selection {self.selection!r}")
        y = (y == self.classes_[1]).astype(int)
        rng = np.random.default_rng(self.random_state)
        V = self.cv
        splitter = StratifiedKFold if self.stratify else KFold
        folds = list(splitter(V, shuffle=True, random_state=int(rng.integers(2**31 - 1))).split(X, y))
        fid = np.empty(len(y), int)
        for v, (_, va) in enumerate(folds):
            fid[va] = v
        cands = self._candidates()
        oof = {c: np.empty(len(y)) for c in cands}
        fold_est = {c: [None] * V for c in cands}
        for v, (tr, va) in enumerate(folds):
            cache = {}
            for c in cands:
                fold_est[c][v], oof[c][va] = self._fit_predict(X[tr], y[tr], X[va], c, rng, cache)
        maps = {c: [None] * V for c in cands}
        pred = {c: np.empty(len(y)) for c in cands}
        if self.nested == "aligned":
            pair = {c: {} for c in cands}
            for u, v in itertools.combinations(range(V), 2):
                idx = np.where((fid != u) & (fid != v))[0]
                iu, iv = folds[u][1], folds[v][1]
                cache = {}
                for c in cands:
                    _, s = self._fit_predict(X[idx], y[idx], np.vstack([X[iu], X[iv]]), c, rng, cache)
                    pair[c][(u, v)] = s[:len(iu)]
                    pair[c][(v, u)] = s[len(iu):]
            for c in cands:
                for v, (_, va) in enumerate(folds):
                    others = [u for u in range(V) if u != v]
                    s = np.concatenate([pair[c][(u, v)] for u in others])
                    yy = np.concatenate([y[folds[u][1]] for u in others])
                    maps[c][v] = make_calibrator(self.calibration).fit(s, yy)
                    pred[c][va] = maps[c][v].predict(oof[c][va])
        else:
            for c in cands:
                for v, (tr, va) in enumerate(folds):
                    maps[c][v] = make_calibrator(self.calibration).fit(oof[c][tr], y[tr])
                    pred[c][va] = maps[c][v].predict(oof[c][va])
        crit_obs = {c: _loss(self.criterion, y, pred[c]) for c in cands}
        crit = {c: float(v.mean()) for c, v in crit_obs.items()}
        best = min(cands, key=crit.get)
        base = cands[0]
        if self.selection == "1se" and best != base:
            d = crit_obs[best] - crit_obs[base]
            if not d.mean() < -d.std(ddof=1) / np.sqrt(len(d)):
                best = base
        elif self.selection == "safe":
            best, self.safe_z_, self.safe_critical_value_ = self._safe_select(cands, crit, crit_obs, fid, rng)
        self.selected_ = best
        self.cv_results_ = {c.label(): crit[c] for c in cands}
        self.candidate_distance_ = float(np.sqrt(np.mean((oof[best] - oof[base]) ** 2)))
        if self.output == "ensemble":
            self.estimators_ = fold_est[best]
            self.calibrators_ = maps[best]
            self.fold_weights_ = np.array([len(va) for _, va in folds], float) / len(y)
        else:
            self.calibrator_ = make_calibrator(self.calibration).fit(oof[best], y)
            self.estimator_, _ = self._fit_predict(X, y, X[:1], best, rng)
        self.n_features_in_ = X.shape[1]
        return self

    def _safe_select(self, cands, crit, crit_obs, fid, rng):
        base = cands[0]
        others = [c for c in cands[1:]]
        n = len(fid)
        D = np.array([crit_obs[c] - crit_obs[base] for c in others])
        Dc = D.copy()
        var = np.zeros(len(others))
        for v in np.unique(fid):
            idx = fid == v
            var += idx.sum() / n * D[:, idx].var(1, ddof=1)
            Dc[:, idx] -= Dc[:, idx].mean(1, keepdims=True)
        tau = self.variance_floor * crit_obs[base].std()
        active = np.flatnonzero(np.maximum(var, tau ** 2) > 0)
        z = np.zeros(len(others))
        z[active] = D[active].mean(1) / np.sqrt(np.maximum(var[active], tau ** 2) / n)
        zdict = {c.label(): float(z[j]) for j, c in enumerate(others)}
        if len(active) == 0:
            return base, zdict, np.inf
        R = np.atleast_2d(np.nan_to_num(np.corrcoef(Dc[active]))) + 1e-9 * np.eye(len(active))
        W = rng.multivariate_normal(np.zeros(len(active)), R, size=20000, method="eigh")
        c_alpha = float(-np.quantile(W.min(1), self.alpha))
        passed = [others[j] for j in active if z[j] < -c_alpha]
        if not passed:
            return base, zdict, c_alpha
        return min(passed, key=crit.get), zdict, c_alpha

    def predict_proba(self, X):
        check_is_fitted(self, "selected_")
        X = check_array(X)
        if self.output == "ensemble":
            p1 = np.average([m.predict(np.clip(e.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6))
                             for e, m in zip(self.estimators_, self.calibrators_)], axis=0,
                            weights=self.fold_weights_)
        else:
            p1 = self.calibrator_.predict(np.clip(self.estimator_.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6))
        return np.c_[1 - p1, p1]

    def predict(self, X, threshold: float = 0.5):
        return self.classes_[(self.predict_proba(X)[:, 1] >= threshold).astype(int)]
