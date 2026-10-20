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
