# Track A progress

Last updated: 15 October 2026 (Asia/Kolkata)

## Implemented

- Synthetic graph, stable propagation, shock scheduling and validation
- SRCS with complete-window filtering, lag decay, mean response and consistency
- Development-only threshold calibration and single-count undirected metrics
- Random, Pearson, lagged-correlation and Granger-style baselines
- Comparison and three robustness experiment runners
- Real-data cache/download, preprocessing, shock detection, ranking and evidence exports
- Seven-page Streamlit dashboard with offline cached-data support
- Tests, Docker, CI, configuration files and submission documentation
- ML-enhanced edge inference using `HistGradientBoostingClassifier`
- Fifteen pair-level features with domain-randomized synthetic training data
- Strict network-seed separation for fitting, threshold calibration and testing
- ML probabilities, permutation feature importance and real-data candidate ranking

## Verification status

`python -m pytest -q`: 33 passed in 3.21 seconds in the clean ML verification environment. The upgraded comparison used 40 domain-randomized training networks, 10 disjoint calibration networks and 10 disjoint test networks. ML achieved F1 0.930 ± 0.081 versus deterministic SRCS 0.838 ± 0.126 on the varied test regimes. Real-data ML analysis succeeded on 1,003 genuine observations and 196 detected events; the highest candidate probability was 0.161, below the synthetic calibration cutoff of 0.301, so no real pair was declared an inferred edge. Streamlit startup and its health endpoint were verified on port 8511. Earlier robustness, PDF and slide checks remain valid; Docker and Streamlit Community Cloud deployment remain unverified.

## Final handoff status

- Feature branch: `feat/complete-track-a`
- Commit author: `Samearth17 <139998526+Samearth17@users.noreply.github.com>`
- Latest deck: `reports/final_submission/Track_A_Presentation_v3.pptx`
- Pull request: https://github.com/ayushmath07/final-year-project/pull/2

## Exact next commands

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m experiments.compare_methods
python -m experiments.run_robustness --seeds 10
python -m experiments.acquire_real_data
python -m experiments.run_real_data
streamlit run app.py --server.headless=true
```
