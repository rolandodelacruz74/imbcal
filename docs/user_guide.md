# imbcal user guide

This guide explains what the estimator does internally and which guarantees its options carry. The API reference is
in the docstrings (`help(imbcal.ImbalanceCalibratedClassifier)`).

## Randomness and common random numbers

`ImbalanceCalibratedClassifier(strategy="augment")` evaluates every candidate `(generator, chi)` by V-fold
cross-validation and, with `nested="aligned"`, by further pairwise fits. Since version 0.4.0 the synthetic draws use
*common random numbers*:

- each generator in `generators` gets a fixed seed, derived deterministically from `random_state` and its index in
  the list (the seeds are reported in `generator_seeds_`);
- every call of the generator, in each fold, in each pairwise fit and in the final refit (`output="refit"`), receives
  a fresh `numpy.random.Generator` seeded with that seed;
- the synthetic sample is therefore a fixed function of the minority sample and of the seed. Two folds with the same
  minority sample get identical synthetic points, and all `chi` values of one generator share the same draws.

This is what the safe-selection theory of the companion paper assumes: the auxiliary randomness is common to all folds
and held fixed, so that each fold predictor is the same function of its training data (Bayle et al., 2020,
footnote 3). For full compliance, the base estimator should also be deterministic given its data, for example with a
fixed `random_state`. Up to version 0.3.1 a single random stream advanced across folds and candidates, so the synthetic
points of each fold came from different draws, and the stream consumed depended on the minority count of each fold.

## Weights of the augmented sample (`weighting`)

With `strategy="augment"`, a candidate `(generator, chi)` adds `m` synthetic minority points (by default
`n_majority - n_minority`) to the training data. The option `weighting` sets their weights.

**`weighting="prevalence"` (default).** Majority points have weight 1, real minority points weight `chi`, and each
synthetic point weight `(1 - chi) n_min / m`. The minority mass stays `n_min`, so every candidate is trained at the
natural prevalence and its target needs no intercept correction. Divided by `n`, the weighted objective equals

```
R_nat + (1 - chi) (n_min / n) * [ mean minority loss on synthetic points
                                  - mean minority loss on real minority points ],
```

where `R_nat` is the ordinary empirical risk at the natural prevalence. This is the debiased-risk form of Schmutz,
Humbert and Mattei (ICLR 2023): the synthetic points enter through a correction term whose expectation is zero only
when the generator reproduces the minority distribution. The prevalence-preserving weights therefore keep the
*target* at the natural prevalence, but they do not remove the generator error. Protection against a poor generator
comes from the selection rule: the recalibrated cross-validated loss, with `chi = 1` (no augmentation) always among
the candidates, and optionally `selection="safe"`.

**`weighting="unit"`.** The ablation of the companion paper: real and synthetic points all have weight 1, and `chi`
indexes the intensity, with `round((1 - chi) m)` synthetic points added (none for `chi = 1`, all `m` for `chi = 0`).
Smaller intensities use the first points of the same seeded draw. The training prevalence shifts with the intensity
and the recalibration absorbs the shift. On the paper's benchmarks this ablation behaved like the prevalence-preserving
version, which indicates that safety came from choosing the intensity by the recalibrated loss rather than from the
weights.

## Safe selection (`selection="safe"`)

SAFE-chi augments only if some candidate beats the natural fit (`chi = 1`) with simultaneous significance, and then
takes the candidate with the smallest criterion among those that pass.

- **Decision test.** For each candidate `j` the per-observation differences
  `D_ji = L(p_j(X_i), Y_i) - L(p_1(X_i), Y_i)` of the recalibrated cross-validated losses give
  `z_j = sqrt(n) mean(D_j) / sd_j`, with `sd_j` the pooled within-fold standard deviation (Bayle, Bayle, Janson and
  Mackey, 2020). This is the V-fold cross-validation-with-confidence test of Lei (2020, *JASA*) for the natural fit, in
  the form studied by Kissel and Lei (arXiv:2211.04958), without Lei's screening of inequalities.
- **Multiple testing.** Many candidates are tested against one reference with a family-wise guarantee for the selected
  one, the structure of the Learn-then-Test literature (Angelopoulos, Bates, Candès, Jordan and Lei). The critical
  value `c_alpha` is the upper `alpha`-quantile of `-min_j G_j`, `G ~ N(0, Gamma)`, with `Gamma` the estimated
  correlation of the fold-centered differences (a max-T, many-to-one critical value). A candidate passes if
  `z_j < -c_alpha`. Diagnostics: `safe_z_` and `safe_critical_value_`.
- **Variance floor.** Each `sd_j` is replaced by `max(sd_j, variance_floor * sd(loss of chi = 1))`. The floor
  protects against spurious significance when the differences are nearly degenerate (for example for `chi` close
  to 1), but it lowers the power for gains smaller than order `n^{-1/2}`. Candidates with zero within-fold variance are
  left out of `Gamma`; if none is left, `c_alpha` is the one-sided normal quantile `z_{1-alpha}` (since 0.4.0; before,
  this edge case gave a critical value close to 0).
- **What is guaranteed.** The guarantee is asymptotic and conditional. Under the conditions of Theorem SAFE-chi (a) of
  the companion paper (folds of equal size assigned independently of the data, `stratify=False`; strictly nested
  recalibration, `nested="aligned"`; auxiliary randomness common to all folds, provided by the fixed generator seeds;
  a fixed grid and a natural fit fixed in advance; and the loss-stability condition of Bayle et al., 2020), the
  probability of selecting a candidate whose fold-average risk is not better than that of `chi = 1` is at most
  `alpha + o(1)`. Loss stability cannot be verified empirically for black-box learners (Luo and Barber,
  arXiv:2405.15107), and the default `stratify=True` departs from the theory. Lei (2020) recommends `alpha` = 0.1 or
  0.2 for discrete model selection.
- **`candidate_distance_`** is the root mean squared distance, on uncalibrated out-of-fold scores, between the selected
  scheme and `chi = 1`. It is a diagnostic related to the expected-loss bound of Proposition 9 of the paper, which uses
  the maximum over candidates of the distance between recalibrated predictions.

Predicted probabilities are clipped to `[1e-6, 1 - 1e-6]`. At this level the constants of the oracle inequalities of
the paper are far from informative, and the theory describes rates only.

## Diagnostics: `majority_exposure`

`majority_exposure(prob_fn, X_min, X_syn)` returns `delta = mean(prob_fn(X_syn)) - mean(prob_fn(X_min))`, the shift
in estimated risk between synthetic and real minority points. It is the plug-in analogue of the population risk shift
`Delta eta` of the companion paper; negative values suggest that the generator moves minority mass towards regions
dominated by the majority class, where the bias costs most.

Treat it as a heuristic diagnostic, not a test. Even the population quantity is only a moderate predictor of harm:
across 20 simulated schemes its rank correlation with the harm after recalibration was about -0.49. The plug-in
version, with an estimated risk function and a finite synthetic sample, is not covered by any theory, and a value near
zero or positive does not rule out local displacements that cancel on average.

## Planning: `rebalancing_cost`

`rebalancing_cost(p, prevalence, auc, n=None, epv=None)` predicts, before any model is fitted, the validation
calibration slope, AUC and log-loss of an unpenalized logistic regression fitted by maximum likelihood, with balanced
weights, after undersampling and after SMOTE, together with the van Houwelingen shrinkage factor.

It is a closed-form planning screen under a Gaussian model (two-class Gaussian mixture in the proportional regime).
Its maximum-likelihood slope predictions agree closely with simulation-based sample-size tools that use only
`(p, prevalence, anticipated AUC)`, in the style of the R package samplesizedev of Pavlou et al. It is optimistic
relative to real data with many dummy predictors, so read it asymmetrically:

- a predicted calibration slope **below 0.9** indicates a design that is too small;
- a predicted slope **above 0.9** does not guarantee adequacy.

For sample-size planning of prediction models, see Riley et al. (2019, *Statistics in Medicine* 38:1276-1296) and
Riley et al. (2025, arXiv:2504.18730).

## Mixed continuous and binary features

`SMOTEGenerator` and `GaussianGenerator` move points continuously in feature space. Applied to a binary or one-hot
encoded column they produce fractional values (SMOTE interpolates between 0 and 1; the smoothed bootstrap adds
Gaussian noise). When `strategy="augment"` uses such a generator on columns whose values are all in {0, 1}, `fit`
emits a `UserWarning`.

Both generators accept `continuous_features`, either column indices or a boolean mask. Only those columns are
perturbed, and the remaining columns are copied from the parent point, so binary columns keep observed values:

```python
from imbcal import ImbalanceCalibratedClassifier, GaussianGenerator, SMOTEGenerator

cont = [0, 1, 2]          # continuous columns; the others are binary
model = ImbalanceCalibratedClassifier(
    estimator, strategy="augment",
    generators=[GaussianGenerator(0.25, continuous_features=cont), SMOTEGenerator(5, continuous_features=cont)],
)
```

SMOTE still finds neighbours on all columns. With the default `continuous_features=None`, all columns are perturbed
and the draws are those of earlier versions for a given random generator. A custom generator can declare
`perturbs_continuously = True` and implement `perturbed_features(n_features)` to take part in the check.
