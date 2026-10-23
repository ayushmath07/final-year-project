# Methodology and reproducibility

The hidden adjacency matrix remains outside all score functions. Each method receives only states and, for SRCS, the event list. Calibration uses labeled development seeds; evaluation uses disjoint test seeds. Undirected metrics use the upper triangle once.

Every experiment writes its parameter configuration, seed policy, raw trial table and summary. A result is reproducible only when the code revision, dependency set, configuration and seeds are retained together.

For real data, the cache has a neighboring provenance JSON file. The analysis metadata marks the run as verified real data only when that file exists. Acquisition errors stop the workflow; no synthetic fallback is labeled as real.
