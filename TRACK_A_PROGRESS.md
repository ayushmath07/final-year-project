# Track A progress

Last updated: 11 October 2026 (Asia/Kolkata)

## Implemented

- Synthetic graph, stable propagation, shock scheduling and validation
- SRCS with complete-window filtering, lag decay, mean response and consistency
- Development-only threshold calibration and single-count undirected metrics
- Random, Pearson, lagged-correlation and Granger-style baselines
- Comparison and three robustness experiment runners
- Real-data cache/download, preprocessing, shock detection, ranking and evidence exports
- Seven-page Streamlit dashboard with offline cached-data support
- Tests, Docker, CI, configuration files and submission documentation

## Verification status

`python -m pytest -q`: 27 passed in 1.35 seconds on the final cached verification run (the clean first run took 45.44 seconds). Full comparison: 10 development plus 10 independent test seeds. Full robustness: 10 seed pairs for each of 12 settings. Real data: acquisition succeeded for 1,003 observations and the analysis detected 196 events. The Streamlit server started on port 8502 and its health endpoint returned successfully. The 19-page PDF was rendered and visually inspected, its text was extracted successfully, and the 10-slide PPTX passed package, layout, chart and import validation. Docker is unavailable on the verification host, so the image build remains unverified. Streamlit Community Cloud deployment also remains unverified because deployment credentials were not available.

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
