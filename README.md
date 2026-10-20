# imbcal

**Calibrated probability estimation under class imbalance.**

```
pip install imbcal
```

[PyPI](https://pypi.org/project/imbcal/) · [Source and issues](https://github.com/rolandodelacruz74/imbcal) · [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23482345.svg)](https://doi.org/10.5281/zenodo.23482345)

Rebalancing (class weights, undersampling, SMOTE) is the default answer to class imbalance, but it targets classification and damages the probabilities that risk prediction needs, even after recalibration. `imbcal` wraps any scikit-learn classifier with pipelines that estimate probabilities at the natural prevalence:

- **`strategy="none"`** (recommended default): fit at the natural prevalence, then recalibrate (Platt or isotonic) on out-of-fold scores.
- **`strategy="augment"`**: *prevalence-preserving augmentation*. Synthetic minority points from any generator receive weights that keep the training prevalence unchanged. Real minority points get weight `chi`, and the `m` synthetic points get `(1 - chi) n_min / m`. The generator and `chi` are chosen by cross-validation of the **recalibrated** loss, and `chi = 1` (no augmentation) is always a candidate, so the selection can fall back to the natural fit.
- **`strategy="weights"`**: balanced weights with recalibration, for comparison.
- **`weighting="unit"`**: the ablation with synthetic points of unit weight, where `chi` indexes the number of synthetic points.
- **`selection="safe"`**: augment only if some candidate beats no augmentation with simultaneous significance (the cross-validation-with-confidence test of Lei, 2020, with a max-T critical value). Under the stated conditions, including loss stability, this controls asymptotically the probability of choosing an augmentation that does not improve on the natural fit.

Synthetic draws use common random numbers: each generator has a fixed seed derived from `random_state`, and the same draws are used in every fold and in the final refit. The [user guide](https://github.com/rolandodelacruz74/imbcal/blob/main/docs/user_guide.md) explains the weights, the safe selection and its assumptions, the diagnostics and the planning tool.

It also provides metrics (log-loss, Brier score, calibration slope and intercept, ECE), paired comparisons over repeated splits (Nadeau–Bengio corrected standard errors and a sign test), diagnostics for synthetic data, and a planning tool:

```python
from imbcal import rebalancing_cost, print_rebalancing_cost
# predicted calibration slope, AUC and log-loss of maximum likelihood and of rebalanced
# logistic regressions, before any data are collected
print_rebalancing_cost(rebalancing_cost(p=36, prevalence=0.092, auc=0.81, epv=2.5))
```

`rebalancing_cost` is a closed-form screen under a Gaussian model and is optimistic with many dummy predictors: a predicted slope below 0.9 flags a design that is too small, but a slope above 0.9 does not guarantee adequacy.

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

Any object with a `sample(X_min, m, rng)` method can be used as a generator, including wrappers around deep tabular generators. With binary or one-hot columns, restrict the built-in generators to the continuous columns, for example `SMOTEGenerator(5, continuous_features=[0, 1, 2])`; `fit` warns when a continuous perturbation acts on a column with values in {0, 1}.

More examples: `examples/quickstart.py` (simulated data, scikit-learn only) and `examples/ist_example.py` (International Stroke Trial).

## Status

Alpha (0.4.0). Binary outcomes only; requires scikit-learn >= 1.6. The API may change before 1.0. Changes are listed in [CHANGELOG.md](https://github.com/rolandodelacruz74/imbcal/blob/main/CHANGELOG.md). Planned additions:
- generators for mixed (continuous and categorical) features beyond `continuous_features`;
- group-wise calibration diagnostics;
- wrappers for deep tabular generators;
- multiclass outcomes.

## Citation

If you use imbcal, please cite it: de la Cruz, R. (2026). *imbcal: Calibrated Probability Estimation under Class Imbalance* [Software]. Zenodo. https://doi.org/10.5281/zenodo.23482345 (this DOI always resolves to the latest version; each release also has its own DOI on Zenodo). The metadata are in `CITATION.cff` (GitHub shows them under "Cite this repository").

## License

MIT.
