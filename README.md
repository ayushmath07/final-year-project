# Inferring Hidden Economic Connections from Financial Shock Propagation

A reproducible Track A prototype that asks whether repeated delayed responses to company-specific shocks contain useful evidence for reconstructing a sparse, static, undirected company network.

The repository contains a complete synthetic benchmark, four baselines, robustness sweeps, a real-price workflow, an offline-capable Streamlit dashboard, tests and final-submission material. Real-data rankings are exploratory statistical evidence. They do **not** prove causality, supplier relationships, ownership, contracts or direction of influence.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m experiments.compare_methods
python -m experiments.run_robustness --seeds 10
python -m experiments.run_real_data
streamlit run app.py
```

The app works offline when `data/real/us_large_caps_2022_2025.csv` is present. To refresh genuine data while online:

```bash
python -m experiments.acquire_real_data
```

If acquisition fails, upload an adjusted-close CSV in the dashboard. The first column must contain dates; remaining columns must contain companies and a broad market index such as `SPY`. The software raises an error rather than substituting synthetic observations.

## Scientific workflow

1. Generate a seeded sparse undirected graph and stable linear propagation process.
2. Inject signed company shocks with configurable separation and complete event windows.
3. Standardize each company series, mapping zero-variance columns to zero.
4. Compute event-conditioned SRCS from the largest lag-decayed, sign-consistent response, its mean and consistency across events.
5. Symmetrize directed evidence without consulting the hidden adjacency matrix.
6. Calibrate a threshold on separate development seeds.
7. Evaluate on independent seeds, counting each undirected pair once.
8. Compare random ranking, Pearson, maximum lagged correlation, pairwise Granger-style predictive gain and SRCS.
9. Repeat experiments across noise, propagation and shock count, reporting mean and standard deviation.
10. For real prices, compute log returns, remove a market-model component, detect cooldown-separated shocks and rank candidate pairs.

## Reproducible outputs

- `reports/results/comparison/`: raw and summary method metrics, rankings, configuration and F1 plot
- `reports/results/robustness/`: raw trials, grouped mean/SD tables, configurations and three robustness figures
- `reports/results/real_data/`: detected events, candidate rankings, event evidence, heatmap, case figures and provenance-aware metadata
- `reports/figures/`: true network, inferred network, propagation and SRCS heatmap
- `reports/black_book.md`: editable manuscript source
- `reports/final_submission/`: rendered DOCX, PDF and validated defense presentation
- `reports/DEMO_SCRIPT.md` and `reports/VIVA_QA.md`: submission support

## Tests and deployment

```bash
python -m pytest -q
streamlit run app.py --server.headless=true
docker build -t hidden-economic-connections .
docker run --rm -p 8501:8501 hidden-economic-connections
```

GitHub Actions repeats unit tests, quick experiment integrations and a Streamlit startup smoke test. For Streamlit Community Cloud, select this repository and branch, set `app.py` as the entry point and deploy. No secrets are required for the packaged dataset. Deployment remains unverified until a maintainer connects the repository in Streamlit Community Cloud.

## Repository map

```text
app.py                         Streamlit interface
src/                           reusable scientific modules
experiments/                   reproducible experiment entry points
configs/                       recorded default parameters
data/synthetic/                packaged synthetic sample
data/real/                     cached genuine adjusted prices and provenance
tests/                         unit and integration tests
reports/                       results and university submission material
```

## Interpretation boundary

The prototype establishes whether the method recovers edges in a controlled synthetic setting. The real-data phase provides ranked candidates and event-level traces for inspection. Common news, sector exposure, nonsynchronous trading, volatility clustering, model misspecification and multiple testing can all produce high scores. Independent economic evidence is required before describing any candidate as a real business connection.
