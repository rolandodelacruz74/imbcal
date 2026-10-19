"""Quickstart: calibrated probabilities under class imbalance on simulated data (scikit-learn only)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from imbcal import GaussianGenerator, ImbalanceCalibratedClassifier, probability_report

rng = np.random.default_rng(0)
n, d = 6000, 5
X = rng.normal(size=(n, d))
logit = -3.5 + 1.5 * X[:, 0] - X[:, 1] ** 2 + 0.8 * X[:, 2] * X[:, 3]
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
print(f"prevalence: {y.mean():.3f}")

base = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)
models = {
    "natural fit + recalibration": ImbalanceCalibratedClassifier(base, strategy="none", random_state=0),
    "balanced weights + recalibration": ImbalanceCalibratedClassifier(base, strategy="weights", random_state=0),
    "safe prevalence-preserving augmentation": ImbalanceCalibratedClassifier(
        base, strategy="augment", selection="safe", chi_grid=(0.0, 0.25, 0.5),
        generators=[GaussianGenerator(0.25), GaussianGenerator(0.5)], random_state=0),
}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    r = probability_report(y_te, model.predict_proba(X_te)[:, 1])
    print(f"{name:42s} log-loss {r['log_loss']:.4f}  Brier {r['brier']:.4f}  AUC {r['auc']:.3f}  "
          f"slope {r['calibration_slope']:.2f}  selected: {model.selected_.label()}")
