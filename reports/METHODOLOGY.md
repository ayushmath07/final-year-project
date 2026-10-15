# Methodology and reproducibility

The hidden adjacency matrix remains outside all score functions. Each method receives only states and, for SRCS, the event list. Calibration uses labeled development seeds; evaluation uses disjoint test seeds. Undirected metrics use the upper triangle once.

Every experiment writes its parameter configuration, seed policy, raw trial table and summary. A result is reproducible only when the code revision, dependency set, configuration and seeds are retained together.

For real data, the cache has a neighboring provenance JSON file. The analysis metadata marks the run as verified real data only when that file exists. Acquisition errors stop the workflow; no synthetic fallback is labeled as real.

## ML-enhanced edge inference

Each unique company pair is represented by 15 ground-truth-free features derived from the observed series and detected event list. These include deterministic SRCS, Pearson and lagged correlation, pairwise Granger-style predictive gain, direction-invariant minimum/maximum response mean and consistency, response asymmetry, event coverage, volatility and standardized-series separation. Only after feature construction is the hidden synthetic adjacency joined as a binary label.

`HistGradientBoostingClassifier` is used because this is a nonlinear tabular classification problem with hundreds to thousands of pair rows, not a large labelled sequence corpus. Balanced class weighting addresses sparse graphs; shallow leaf counts, minimum leaf size and L2 regularization constrain complexity.

The synthetic training corpus uses deterministic domain randomization across company count, density, persistence, propagation, noise, shock size and shock count. Splitting occurs by complete simulation seed:

- seeds 0–39: model fitting;
- seeds 100–109: probability-threshold calibration and permutation interpretation;
- seeds 200–209: final comparison only.

No pair from a graph can occur in more than one split. Baselines and ML are evaluated on the same test networks. Real-data inference is an out-of-domain exploratory application and is not reported as supervised real-world accuracy because independent real edge labels are absent.
