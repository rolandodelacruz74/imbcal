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
