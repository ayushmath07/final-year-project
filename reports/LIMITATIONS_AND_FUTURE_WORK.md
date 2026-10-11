# Limitations ethics and future work

- The synthetic network is static, binary and undirected.
- The propagation equation is linear and uses Gaussian noise.
- Daily returns can hide intraday ordering and mix simultaneous events.
- A single-factor market model leaves sector and macroeconomic confounding.
- Detected shocks are statistical extremes, not verified corporate announcements.
- Candidate rankings do not establish causality or a contractual relationship.
- Small sparse graphs make F1 sensitive to a few edge decisions.
- Hyperparameters were chosen for an interpretable prototype, not optimized for trading.

Future work may add sector-factor residualization, uncertainty intervals, multiple-testing control, directed or time-varying networks and external relationship labels. Deep learning, graph neural networks and complex Bayesian models remain outside Track A.
