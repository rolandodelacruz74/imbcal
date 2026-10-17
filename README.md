# imbcal

**Calibrated probability estimation under class imbalance.**

Rebalancing (class weights, undersampling, SMOTE) is the default answer to class imbalance, but it targets classification and damages the probabilities that risk prediction needs, even after recalibration. `imbcal` wraps any scikit-learn classifier with pipelines that estimate probabilities at the natural prevalence:

- **`strategy="none"`** (recommended default): fit at the natural prevalence, then recalibrate (Platt or isotonic) on out-of-fold scores.
- **`strategy="augment"`**: *prevalence-preserving augmentation*. Synthetic minority points from any generator receive weights that keep the training prevalence unchanged. Real minority points get weight `chi`, and the `m` synthetic points get `(1 - chi) n_min / m`. The generator and `chi` are chosen by cross-validation of the **recalibrated** loss, and `chi = 1` (no augmentation) is always a candidate, so the selection can fall back to the natural fit.
- **`strategy="weights"`**: balanced weights with recalibration, for comparison.

It also provides metrics (log-loss, Brier score, calibration slope and intercept, ECE), paired comparisons with Nadeau–Bengio corrected standard errors, and diagnostics for synthetic data.

## Example

```python
from sklearn.ensemble import RandomForestClassifier
from imbcal import ImbalanceCalibratedClassifier, GaussianGenerator, SMOTEGenerator, probability_report

model = ImbalanceCalibratedClassifier(
    RandomForestClassifier(min_samples_leaf=10),
    strategy="augment",
    generators=[GaussianGenerator(0.25), SMOTEGenerator(k="sqrt")],
    calibration="platt", selection="1se", random_state=0,
).fit(X_train, y_train)

print(model.selected_.label(), model.candidate_distance_)
print(probability_report(y_test, model.predict_proba(X_test)[:, 1]))
```

Any object with a `sample(X_min, m, rng)` method can be used as a generator, including wrappers around deep tabular generators.

## Status

Pre-release (0.1.0.dev0). The API may change. Planned additions:
- the `rebalancing_cost` predictor (asymptotic cost of rebalancing for logistic regression);
- truncation of candidates for very rare events;
- further generators;
- a benchmarking module.

## License

MIT.
