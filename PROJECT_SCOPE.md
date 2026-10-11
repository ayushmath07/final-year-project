# Track A scope

## Title

**Inferring Hidden Economic Connections from Financial Shock Propagation: A Synthetic and Real-Data Prototype**

## Research question

Can repeated lagged responses after company-specific financial shocks provide useful evidence for reconstructing hidden economic connections between companies?

## Included

- Static sparse binary undirected synthetic networks with known ground truth
- Stable linear shock propagation, noise and deterministic seeds
- Event-conditioned SRCS, threshold calibration and independent evaluation
- Random, Pearson, maximum lagged-correlation and Granger-style baselines
- Robustness experiments for noise, propagation and number of shocks
- A small real adjusted-price pipeline with market residualization and event evidence
- Streamlit dashboard, tests, Docker and CI configuration
- Black-book, demo and viva material

## Excluded Track B work

- Dynamic or directed latent networks
- Deep learning, graph neural networks and Bayesian network reconstruction
- Production trading signals, forecasts or automated decisions
- Claims that market co-response proves causality or a commercial relationship

## Ground-truth separation rule

Inference functions receive observations and events only. Development adjacency matrices may be used solely to select a threshold. Test adjacency matrices are exposed only to metric functions after inference. Real-data analysis has no ground-truth adjacency and therefore exports rankings rather than recovery metrics.

## Completion criteria

The essential Track A implementation is complete when the test suite passes, full comparison and robustness outputs are regenerated, genuine price data or an explicit acquisition dependency is documented, the offline dashboard starts, and manuscript claims match recorded results.
