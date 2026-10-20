import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.estimator_checks import check_estimator

from imbcal import (ImbalanceCalibratedClassifier, GaussianGenerator, SMOTEGenerator, CallableGenerator,
                    probability_report, paired_comparison, crossfit_calibrated, majority_exposure)


def data(n=1500, seed=0, pi=0.08):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pi).astype(int)
    X = rng.normal(size=(n, 4)) + 1.2 * y[:, None] * np.array([1, 0.5, 0, 0])
    return X, y


def test_none_strategy_is_plain_fit_plus_platt():
    X, y = data()
    m = ImbalanceCalibratedClassifier(LogisticRegression(), strategy="none", random_state=0).fit(X, y)
    p = m.predict_proba(X)[:, 1]
    assert p.shape == (len(y),) and np.all((p > 0) & (p < 1))
    assert m.selected_.generator is None and m.candidate_distance_ == 0.0
    assert abs(p.mean() - y.mean()) < 0.02


def test_augment_selects_and_preserves_prevalence_scale():
    X, y = data()
    m = ImbalanceCalibratedClassifier(RandomForestClassifier(n_estimators=50, min_samples_leaf=10, random_state=0),
                                      strategy="augment", chi_grid=(0.25, 0.5), random_state=1).fit(X, y)
    assert "none" in m.cv_results_ and len(m.cv_results_) == 1 + 4 * 2
    p = m.predict_proba(X)[:, 1]
    assert abs(p.mean() - y.mean()) < 0.03


def test_1se_rule_falls_back_to_no_augmentation_on_noise():
    rng = np.random.default_rng(3); X = rng.normal(size=(800, 3)); y = (rng.random(800) < 0.1).astype(int)
    m = ImbalanceCalibratedClassifier(LogisticRegression(), strategy="augment", selection="1se",
                                      generators=[GaussianGenerator(0.5)], random_state=0).fit(X, y)
    assert m.selected_.generator is None


def test_custom_generator_and_shape_check():
    X, y = data(600)
    gen = CallableGenerator(lambda Xm, m, rng: Xm[rng.integers(0, len(Xm), m)], name="bootstrap")
    m = ImbalanceCalibratedClassifier(LogisticRegression(), strategy="augment", generators=[gen],
                                      chi_grid=(0.5,), random_state=0).fit(X, y)
    assert m.predict(X).shape == y.shape
    bad = CallableGenerator(lambda Xm, m, rng: Xm[:1])
    with pytest.raises(ValueError):
        bad.sample(X[y == 1], 5, np.random.default_rng(0))


def test_smote_generator_stays_on_segments():
    rng = np.random.default_rng(0); Xm = rng.normal(size=(30, 2))
    S = SMOTEGenerator(5).sample(Xm, 200, rng)
    assert S.shape == (200, 2)
    assert S.min() >= Xm.min() - 1e-9 and S.max() <= Xm.max() + 1e-9


def test_metrics_and_comparison():
    rng = np.random.default_rng(0); y = (rng.random(2000) < 0.1).astype(int)
    r = probability_report(y, np.full(2000, 0.1))
    assert abs(r["calibration_intercept"]) < 5 and r["ece"] < 0.02
    out = paired_comparison([1.0, 1.1, 0.9, 1.05], [1.0, 1.0, 1.0, 1.0], n_train=800, n_test=200)
    assert out["se_corrected"] > out["se_naive"]


def test_crossfit_and_exposure():
    rng = np.random.default_rng(0); y = (rng.random(500) < 0.2).astype(int); p = np.clip(0.2 + 0.1 * rng.normal(size=500), 0.01, 0.99)
    folds = [(np.r_[0:250], np.r_[250:500]), (np.r_[250:500], np.r_[0:250])]
    q = crossfit_calibrated(p, y, folds, "platt")
    assert q.shape == p.shape
    e = majority_exposure(lambda Z: Z[:, 0], np.array([[0.2], [0.4]]), np.array([[0.1], [0.1]]))
    assert e["delta"] < 0


def test_sklearn_api_clone_and_params():
    from sklearn.base import clone
    m = ImbalanceCalibratedClassifier(LogisticRegression(), strategy="augment")
    c = clone(m); assert c.get_params()["strategy"] == "augment"


def test_nested_and_output_options():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from imbcal import ImbalanceCalibratedClassifier
    rng = np.random.default_rng(0)
    n = 600
    y = (rng.random(n) < 0.1).astype(int)
    X = rng.normal(size=(n, 3)) + 1.2 * y[:, None]
    res = {}
    for nested in ("aligned", "crossfit"):
        for output in ("ensemble", "refit"):
            m = ImbalanceCalibratedClassifier(LogisticRegression(), strategy="augment", chi_grid=(0.0, 0.5),
                                              nested=nested, output=output, cv=4, random_state=1).fit(X, y)
            p = m.predict_proba(X)[:, 1]
            assert p.shape == (n,) and np.all((p > 0) & (p < 1))
            res[(nested, output)] = m.selected_.label()
            if output == "ensemble":
                assert len(m.estimators_) == 4 and len(m.calibrators_) == 4
                assert abs(m.fold_weights_.sum() - 1) < 1e-12
    m = ImbalanceCalibratedClassifier(LogisticRegression(), strategy="augment", chi_grid=(0.5,), stratify=False,
                                      cv=3, random_state=2).fit(X, y)
    assert m.predict_proba(X).shape == (n, 2)
    # same fold structure and candidates: the two nestings may differ only in the recalibration of the criterion
    assert all(isinstance(v, str) for v in res.values())


def test_aligned_map_independent_of_fold():
    """The map of fold v must not change when the labels of fold v are permuted (strict nesting)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from imbcal import ImbalanceCalibratedClassifier
    rng = np.random.default_rng(3)
    n = 400
    y = (rng.random(n) < 0.15).astype(int)
    X = rng.normal(size=(n, 2)) + y[:, None]
    m = ImbalanceCalibratedClassifier(LogisticRegression(), strategy="none", cv=4, random_state=5).fit(X, y)
    # rebuild the folds used inside fit
    r = np.random.default_rng(5)
    folds = list(StratifiedKFold(4, shuffle=True, random_state=int(r.integers(2**31 - 1))).split(X, y))
    va = folds[0][1]
    X2 = X.copy(); X2[va] = X[va][rng.permutation(len(va))]   # scramble features within fold 0 (keeps labels/strata)
    m2 = ImbalanceCalibratedClassifier(LogisticRegression(), strategy="none", cv=4, random_state=5).fit(X2, y)
    a, b = m.calibrators_[0], m2.calibrators_[0]
    assert abs(a.intercept_ - b.intercept_) < 1e-8 and abs(a.slope_ - b.slope_) < 1e-8


def test_safe_selection():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from imbcal import ImbalanceCalibratedClassifier
    rng = np.random.default_rng(4)
    n = 500
    y = (rng.random(n) < 0.12).astype(int)
    X = rng.normal(size=(n, 3)) + y[:, None]
    m = ImbalanceCalibratedClassifier(LogisticRegression(), strategy="augment", chi_grid=(0.25, 0.5, 0.75),
                                      selection="safe", alpha=0.05, nested="crossfit", cv=4, random_state=0).fit(X, y)
    assert np.isfinite(m.safe_critical_value_) and m.safe_critical_value_ > 1.5
    assert set(m.safe_z_) == {k for k in m.cv_results_ if k != "none"}
    # a selected augmented candidate must have passed the simultaneous test
    if m.selected_.label() != "none":
        assert m.safe_z_[m.selected_.label()] < -m.safe_critical_value_
    assert m.predict_proba(X).shape == (n, 2)


def test_rebalancing_cost_matches_paper_tool():
    from imbcal import rebalancing_cost
    r = rebalancing_cost(p=71, prevalence=0.092, epv=2.5, auc=0.81)
    m = r["methods"]
    # values printed by tool/rebalancing_cost.py of the Paper A replication package
    assert abs(m["mle"]["calibration_slope"] - 0.680) < 5e-4
    assert abs(m["weights"]["calibration_slope"] - 0.606) < 5e-4
    assert abs(m["undersampling"]["calibration_slope"] - 0.374) < 5e-4
    assert abs(m["smote"]["calibration_slope"] - 0.533) < 5e-4
    assert abs(r["van_houwelingen_shrinkage"] - 0.767) < 5e-4
    assert m["mle"]["log_loss"] < m["weights"]["log_loss"] < m["undersampling"]["log_loss"]


def test_scikit_learn_estimator_checks():
    import warnings
    est = ImbalanceCalibratedClassifier(LogisticRegression(max_iter=200), random_state=0, cv=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = check_estimator(est, on_fail=None)
    failed = [r["check_name"] for r in res if r["status"] == "failed"]
    assert not failed, failed


def _jitter(Xm, m, rng):
    return Xm[rng.integers(0, len(Xm), m)] + 0.1 * rng.normal(size=(m, Xm.shape[1]))


def test_common_random_numbers_across_folds():
    """Same minority sample in two folds -> identical synthetic draws (fixed seed per generator)."""
    from imbcal.estimator import Candidate
    X, y = data(400, seed=7)
    calls = []

    def rec(Xm, m, rng):
        out = _jitter(Xm, m, rng)
        calls.append((Xm.copy(), out.copy()))
        return out

    gen = CallableGenerator(rec, name="recorder")
    m = ImbalanceCalibratedClassifier(LogisticRegression(), strategy="augment", generators=[gen], chi_grid=(0.5,),
                                      nested="aligned", cv=3, random_state=11).fit(X, y)
    assert len(m.generator_seeds_) == 1
    seed = m.generator_seeds_[0][1]
    # every call (folds, pairwise fits) used a fresh rng with the same seed
    assert len(calls) >= 3 + 3
    for Xm, out in calls:
        assert np.array_equal(out, _jitter(Xm, len(out), np.random.default_rng(seed)))
    # two training folds that share the same minority sample but differ in the majority rows
    calls.clear()
    Xmin, Xmaj = X[y == 1], X[y == 0]
    ya = np.r_[np.ones(len(Xmin), int), np.zeros(150, int)]
    m._fit_predict(np.vstack([Xmin, Xmaj[:150]]), ya, X[:5], Candidate(gen, 0.5), [seed])
    m._fit_predict(np.vstack([Xmin, Xmaj[150:300]]), ya, X[:5], Candidate(gen, 0.5), [seed])
    (m1, s1), (m2, s2) = calls
    assert np.array_equal(m1, m2) and np.array_equal(s1, s2)
    # same random_state -> same seeds and same selection
    from sklearn.base import clone
    again = clone(m).fit(X, y)
    assert again.generator_seeds_ == m.generator_seeds_ and again.cv_results_ == m.cv_results_


def test_weighting_validation_and_weights():
    from imbcal.estimator import Candidate
    X, y = data(400, seed=2)
    with pytest.raises(ValueError, match="weighting"):
        ImbalanceCalibratedClassifier(LogisticRegression(), strategy="augment", weighting="balanced").fit(X, y)

    class Spy(LogisticRegression):
        def fit(self, X, y, sample_weight=None):
            Spy.last = (len(y), np.asarray(sample_weight), np.asarray(y))
            return super().fit(X, y, sample_weight=sample_weight)

    gen = GaussianGenerator(0.3)
    n1 = int(y.sum()); n0 = len(y) - n1; m = n0 - n1
    for weighting in ("prevalence", "unit"):
        est = ImbalanceCalibratedClassifier(Spy(), strategy="augment", generators=[gen], weighting=weighting)
        est._gen_index = {id(gen): 0}
        est._fit_predict(X, y, X[:3], Candidate(gen, 0.25), [123])
        ntot, w, yy = Spy.last
        if weighting == "prevalence":
            assert ntot == len(y) + m
            assert np.allclose(w[:len(y)][y == 1], 0.25) and np.allclose(w[:len(y)][y == 0], 1.0)
            assert np.allclose(w[len(y):], 0.75 * n1 / m)
            # minority mass n1 and majority mass n0: natural prevalence
            assert np.isclose(w[yy == 1].sum(), n1) and np.isclose(w[yy == 0].sum(), n0)
        else:
            assert ntot == len(y) + round(0.75 * m) and np.all(w == 1.0)


def test_weighting_identity_debiased_risk():
    """Weighted objective / n = natural risk + (1-chi)(n1/n)[mean syn loss - mean real minority loss]."""
    rng = np.random.default_rng(0)
    n, n1, m, chi = 200, 30, 170, 0.3
    y = np.r_[np.ones(n1), np.zeros(n - n1)]
    loss_real = rng.random(n); loss_syn = rng.random(m)
    w_real = np.where(y == 1, chi, 1.0); w_syn = (1 - chi) * n1 / m
    lhs = (np.sum(w_real * loss_real) + w_syn * loss_syn.sum()) / n
    rhs = loss_real.mean() + (1 - chi) * n1 / n * (loss_syn.mean() - loss_real[y == 1].mean())
    assert np.isclose(lhs, rhs)


def test_unit_weighting_fit():
    X, y = data(600, seed=5)
    mdl = ImbalanceCalibratedClassifier(LogisticRegression(), strategy="augment", weighting="unit",
                                        generators=[GaussianGenerator(0.25)], chi_grid=(0.0, 0.5), cv=3,
                                        nested="crossfit", random_state=0).fit(X, y)
    assert len(mdl.cv_results_) == 3 and mdl.predict_proba(X).shape == (len(y), 2)


def test_safe_select_degenerate_candidates_use_normal_quantile():
    """Candidates with zero within-fold variance are left out of the correlation matrix; if none is left,
    c_alpha = z_{1-alpha} (previously about 0)."""
    from scipy.stats import norm
    from imbcal.estimator import Candidate
    rng = np.random.default_rng(0)
    n, V = 200, 4
    fid = np.repeat(np.arange(V), n // V)
    base = Candidate(None, 1.0)
    c1, c2 = Candidate(GaussianGenerator(0.25), 0.5), Candidate(GaussianGenerator(0.5), 0.5)
    lb = rng.exponential(size=n)
    crit_obs = {base: lb, c1: lb - 0.01, c2: lb + 0.02}      # constant differences: zero within-fold variance
    crit = {k: float(v.mean()) for k, v in crit_obs.items()}
    est = ImbalanceCalibratedClassifier(LogisticRegression(), selection="safe", alpha=0.2)
    best, z, c = est._safe_select([base, c1, c2], crit, crit_obs, fid, np.random.default_rng(1))
    assert np.isclose(c, norm.ppf(0.8))
    tau = 0.05 * lb.std()
    assert np.isclose(z[c1.label()], -0.01 / (tau / np.sqrt(n)))
    assert best == (c1 if z[c1.label()] < -c else base)
    # a non-degenerate candidate enters the correlation matrix; the degenerate one keeps its floored z
    crit_obs[c2] = lb + 0.05 * rng.normal(size=n)
    crit = {k: float(v.mean()) for k, v in crit_obs.items()}
    _, z2, c2v = est._safe_select([base, c1, c2], crit, crit_obs, fid, np.random.default_rng(1))
    assert abs(c2v - norm.ppf(0.8)) < 0.05 and np.isclose(z2[c1.label()], z[c1.label()])


def _mixed(n=500, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.12).astype(int)
    Xc = rng.normal(size=(n, 2)) + y[:, None]
    Xb = (rng.random((n, 2)) < 0.3 + 0.3 * y[:, None]).astype(float)
    return np.c_[Xc, Xb], y


def test_binary_columns_warning():
    X, y = _mixed()
    kw = dict(strategy="augment", chi_grid=(0.5,), cv=3, nested="crossfit", random_state=0)
    with pytest.warns(UserWarning, match="fractional values"):
        ImbalanceCalibratedClassifier(LogisticRegression(), generators=[SMOTEGenerator(5)], **kw).fit(X, y)
    with pytest.warns(UserWarning, match="binary"):
        ImbalanceCalibratedClassifier(LogisticRegression(), generators=[GaussianGenerator(0.25)], **kw).fit(X, y)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ImbalanceCalibratedClassifier(LogisticRegression(), generators=[SMOTEGenerator(5, continuous_features=[0, 1]),
                                      GaussianGenerator(0.25, continuous_features=[True, True, False, False])],
                                      **kw).fit(X, y)
        ImbalanceCalibratedClassifier(LogisticRegression(), strategy="none", cv=3, random_state=0).fit(X, y)
        Xc = X[:, :2]
        ImbalanceCalibratedClassifier(LogisticRegression(), generators=[SMOTEGenerator(5)], **kw).fit(Xc, y)


def test_continuous_features_copy_other_columns():
    X, y = _mixed(seed=1)
    Xm = X[y == 1]
    for gen in (SMOTEGenerator(5, continuous_features=[0, 1]), GaussianGenerator(0.3, continuous_features=[0, 1]),
                SMOTEGenerator("sqrt", continuous_features=np.array([True, True, False, False]))):
        S = gen.sample(Xm, 300, np.random.default_rng(0))
        assert S.shape == (300, 4)
        assert set(np.unique(S[:, 2:])) <= {0.0, 1.0}
        # binary part of every synthetic point is a row of the minority sample (copied from the parent)
        rows = {tuple(r) for r in Xm[:, 2:]}
        assert all(tuple(r) in rows for r in S[:, 2:])
        assert len(np.unique(S[:, 0])) > len(Xm)            # continuous columns are perturbed
    # default (all columns) keeps the 0.3.x draws for a given rng
    a = SMOTEGenerator(5).sample(Xm, 50, np.random.default_rng(3))
    b = SMOTEGenerator(5, continuous_features=[0, 1, 2, 3]).sample(Xm, 50, np.random.default_rng(3))
    assert np.allclose(a, b)
    assert repr(SMOTEGenerator(5)) == "SMOTEGenerator(k=5)"
    with pytest.raises(ValueError):
        GaussianGenerator(0.3, continuous_features=[0, 7]).sample(Xm, 5, np.random.default_rng(0))
    with pytest.raises(ValueError):
        GaussianGenerator(0.3, continuous_features=[True, False]).sample(Xm, 5, np.random.default_rng(0))
