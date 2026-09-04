# Final Year Project: Hidden Economic Connections

Step 1 of **Inferring Hidden Economic Connections from Financial Shock
Propagation** is complete here: scope freeze, a reproducible synthetic network,
and a first shock-propagation experiment.

The project asks whether repeated delayed reactions to company-specific shocks
can help recover otherwise hidden economic connections.  This initial version
does **not** infer connections yet; that is the next SRCS-inference milestone.

## What Step 1 includes

- frozen problem statement, objectives, assumptions, and exclusions
- an 8-company sparse, undirected hidden network with known ground truth
- row-normalized propagation using
  `X[t+1] = persistence * X[t] + propagation * A_normalized * X[t] + shock[t+1] + noise[t+1]`
- deterministic random seeds, saved simulation data, event list, and figures
- unit tests for graph validity, normalization, reproducibility, and output shape

## Run it

```powershell
python -m pip install -r requirements.txt
python scripts/run_step_1.py
python -m pytest -q
```

The run regenerates these files:

- `data/synthetic/step_1_sample.npz`
- `data/synthetic/step_1_events.csv`
- `reports/figures/system_architecture.png`
- `reports/figures/true_network.png`
- `reports/figures/shock_propagation.png`

## Project boundary

Synthetic data is the source of ground truth.  Real stock data will later be
used only to rank candidate connections, not to claim causality or prove a
commercial relationship.

See [PROJECT_SCOPE.md](PROJECT_SCOPE.md) for the agreed Step 1 scope.
