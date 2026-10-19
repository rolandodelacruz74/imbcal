# imbcal

**Calibrated probability estimation under class imbalance.**

```
pip install imbcal
```

Rebalancing (class weights, undersampling, SMOTE) is the default answer to class imbalance, but it targets classification and damages the probabilities that risk prediction needs, even after recalibration. `imbcal` wraps any scikit-learn classifier with pipelines that estimate probabilities at the natural prevalence:

- **`strategy="none"`** (recommended default): fit at the natural prevalence, then recalibrate (Platt or isotonic) on out-of-fold scores.
- **`strategy="augment"`**: *prevalence-preserving augmentation*. Synthetic minority points from any generator receive weights that keep the training prevalence unchanged. Real minority points get weight `chi`, and the `m` synthetic points get `(1 - chi) n_min / m`. The generator and `chi` are chosen by cross-validation of the **recalibrated** loss, and `chi = 1` (no augmentation) is always a candidate, so the selection can fall back to the natural fit.
- **`strategy="weights"`**: balanced weights with recalibration, for comparison.
- **`selection="safe"`**: augment only if some candidate beats no augmentation with simultaneous significance (cross-validation with confidence and a max-T critical value), which controls the probability of choosing a harmful augmentation.

It also provides metrics (log-loss, Brier score, calibration slope and intercept, ECE), paired comparisons over repeated splits (Nadeau–Bengio corrected standard errors and a sign test), diagnostics for synthetic data, and a planning tool:

```python
from imbcal import rebalancing_cost, print_rebalancing_cost
# predicted calibration slope, AUC and log-loss of maximum likelihood and of rebalanced
# logistic regressions, before any data are collected
print_rebalancing_cost(rebalancing_cost(p=36, prevalence=0.092, auc=0.81, epv=2.5))
```

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

More examples: `examples/quickstart.py` (simulated data, scikit-learn only) and `examples/ist_example.py` (International Stroke Trial).

## Status

Alpha (0.3.0). Binary outcomes only; requires scikit-learn >= 1.6. The API may change before 1.0. Planned additions:
- generators for mixed (continuous and categorical) features;
- group-wise calibration diagnostics;
- wrappers for deep tabular generators;
- multiclass outcomes.

## Citation

If you use imbcal, please cite it; the metadata are in `CITATION.cff` (GitHub shows them under "Cite this repository").

## License

MIT.
