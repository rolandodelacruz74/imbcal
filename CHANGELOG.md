# Changelog

## 0.4.0 (2026-10-20)

### Changed
- **Common random numbers across folds.** Each generator gets a fixed seed derived from `random_state` and its index
  in `generators`, and every call (cross-validation folds, pairwise fits of `nested="aligned"`, final refit) uses a
  fresh `numpy.random.Generator` with that seed. The synthetic draws are a fixed function of the minority sample and
  the seed, as the SAFE-chi theory assumes. Previously one random stream advanced across folds and candidates, so
  results for a given `random_state` differ from 0.3.x when `strategy="augment"`. The fold assignment is unchanged.
  New attribute `generator_seeds_`.
- **`selection="safe"`:** candidates with zero within-fold variance are left out of the correlation matrix of the
  max-T critical value; if none is left, the critical value is the one-sided normal quantile `z_{1-alpha}` (it was
  close to 0 in this edge case).

### Added
- `weighting` parameter: `"prevalence"` (default, the previous prevalence-preserving weights) or `"unit"` (real and
  synthetic points of weight 1, with `chi` indexing the number of synthetic points, as in the paper's ablation).
- `continuous_features` in `SMOTEGenerator` and `GaussianGenerator`: only those columns are perturbed and the others
  are copied from the parent point.
- `UserWarning` when a generator that perturbs features continuously acts on columns whose values are all in {0, 1}.
- `docs/user_guide.md`.

### Documentation
- Weights: the debiased-risk identity (Schmutz, Humbert and Mattei, ICLR 2023); the weights keep the natural-prevalence
  target but do not remove the generator error.
- Safe selection: attribution to Lei (2020) and Kissel and Lei (arXiv:2211.04958) and to the Learn-then-Test
  literature; the guarantee is asymptotic and conditional on loss stability (Bayle et al., 2020), which cannot be
  verified for black-box learners (Luo and Barber, arXiv:2405.15107); the variance floor lowers power for gains of
  order smaller than `n^{-1/2}`; corrected description of `candidate_distance_` (Proposition 9, expected loss,
  uncalibrated scores).
- `majority_exposure` described as a heuristic diagnostic; `rebalancing_cost` as a closed-form Gaussian-model planning
  screen (slope below 0.9 flags a design that is too small; above 0.9 is no guarantee). Numerics unchanged.

## 0.3.1 (2026-10-19)
- Repository URLs in the package metadata and `CITATION.cff`; Zenodo concept DOI.

## 0.3.0 (2026-10-19)
- First PyPI release; scikit-learn >= 1.6, SPDX license metadata.
