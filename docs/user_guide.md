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
