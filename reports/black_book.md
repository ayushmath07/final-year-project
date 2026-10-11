# Inferring Hidden Economic Connections from Financial Shock Propagation

## A Synthetic and Real Data Prototype

**Student:** [STUDENT NAME AND ID]

**Guide:** [GUIDE NAME]

**Department and institution:** [INSTITUTIONAL DETAILS]

**Academic year:** [ACADEMIC YEAR]

> Submission certificates, declarations, signatures and institution-specific formatting remain placeholders pending official details.

## Abstract

Economic relationships between companies are often incomplete, private or difficult to observe directly. This project tests a narrow hypothesis: repeated delayed responses after company-specific financial shocks may contain useful evidence about an unobserved connection network. The prototype creates sparse undirected synthetic networks, simulates stable shock propagation and introduces the Shock-Response Connectivity Score (SRCS). SRCS combines lag-decayed response magnitude with consistency across source-company events. Thresholds are calibrated on development simulations and evaluated on independent test seeds. Four reference methods provide context: random ranking, contemporaneous Pearson correlation, maximum lagged correlation and a pairwise Granger-style predictive-gain score. Robustness experiments vary noise, propagation strength and the number of shocks. A separate real-price pipeline removes a broad market component, detects large abnormal returns and ranks candidate connections without claiming causality. The implementation includes reproducible scripts, raw result tables, tests and an offline-capable Streamlit dashboard.

## 1 Introduction

Networks offer a compact representation of systems in which entities interact. In an economic network, nodes may represent firms and edges may represent supply, ownership, financing, technological or other dependencies. Direct edge data may be incomplete or unavailable, motivating network reconstruction from observed behavior. Financial network reconstruction is an established problem precisely because private interconnections limit direct measurement [1].

This work studies a controlled prototype rather than a general discovery system. Its central question is whether conditioning on repeated source-company shocks adds useful information beyond unconditional co-movement. Synthetic data provides the only ground truth. Real prices support an exploratory demonstration and produce ranked candidates only.

### 1.1 Objectives

The project aims to implement an end-to-end reproducible benchmark, compare SRCS with simple alternatives, measure robustness and expose the evidence through a dashboard suitable for a university demonstration.

### 1.2 Contributions

The implementation contributes an event-conditioned score, explicit development/test separation, edge-level ranking metrics, three repeated robustness studies, a provenance-aware real-data path and an auditable event-evidence export.

## 2 Literature Review

Network science formalizes systems as nodes and edges and studies both topology and processes that unfold over it [2]. Reconstruction methods estimate missing structure from partial observations. Anand and co-authors survey reconstruction in economic and financial systems and emphasize that different assumptions can produce different inferred networks [1].

Event studies measure price behavior around identified events and motivate careful separation of expected and abnormal returns [3]. The present prototype borrows the event-conditioning idea but does not estimate announcement effects or make causal claims. The market model removes a linear common component before shock detection.

Granger's predictive definition asks whether past values from one series improve predictions of another [4]. The baseline implemented here uses the relative residual-sum-of-squares improvement from adding lagged source values to an autoregression of the target. It is described as “Granger-style” because the prototype records a predictive score rather than a full inferential test with diagnostic assumptions.

Pearson correlation and lagged correlation are useful references but can reflect shared factors, persistence or simultaneous news. SRCS attempts to focus the comparison on repeated source-specific extreme observations.

## 3 Problem Definition and Mathematical Background

Let an unobserved network contain \(n\) companies and adjacency matrix \(A\), where \(A_{ij}=A_{ji}\), \(A_{ii}=0\) and \(A_{ij}\in\{0,1\}\). Synthetic company states evolve as

\[
x_{t+1}=\alpha x_t+\beta D^{-1}Ax_t+s_{t+1}+\epsilon_{t+1},
\]

where \(\alpha\) is persistence, \(\beta\) is propagation strength, \(D^{-1}A\) is the row-normalized adjacency matrix, \(s_t\) contains sparse signed shocks and \(\epsilon_t\) is Gaussian noise. The implementation requires \(\alpha+\beta<1\) as a conservative stability condition.

For a source event \(e=(t,i,m)\), responder \(j\), maximum lag \(L\), standardized observation \(z\), and decay \(\lambda\), event response strength is

\[
r_{e,i\rightarrow j}=\max_{1\leq \ell\leq L}\left[e^{-\lambda(\ell-1)}\max(0,\operatorname{sign}(m)z_{t+\ell,j})\right].
\]

The directed score multiplies mean response by the fraction of event responses above threshold \(\tau_r\):

\[
Q_{i\rightarrow j}=\overline{r}_{i\rightarrow j}\cdot \frac{1}{|E_i|}\sum_{e\in E_i}\mathbf{1}[r_e\geq\tau_r].
\]

The undirected SRCS is \(Q_{ij}=(Q_{i\rightarrow j}+Q_{j\rightarrow i})/2\). A calibrated score threshold produces a binary inferred adjacency. Calibration accesses only development truth; inference itself receives observations and events, not \(A\).

## 4 Proposed System Design

The system separates scientific computation from presentation. Reusable modules implement simulation, preprocessing, detection, scoring, baselines, thresholding and evaluation. Experiment entry points store raw trials, summaries, configurations and figures. The dashboard calls these modules rather than duplicating equations.

```mermaid
flowchart LR
  C[Configuration and seed] --> S[Synthetic simulator]
  S --> O[Observations and event list]
  O --> M[SRCS and baseline scores]
  M --> T[Development threshold calibration]
  T --> I[Test network inference]
  I --> E[Metrics tables and figures]
  P[Cached or uploaded prices] --> R[Returns and market residuals]
  R --> D[Shock detection]
  D --> Q[Candidate ranking and event evidence]
```

## 5 Methodology

Synthetic development and test seeds never overlap. Every method receives the same observation matrix for a seed. Each method calibrates its own threshold on the same development simulations, preventing the arbitrary reuse of one method's score scale. Precision, recall and F1 count the strict upper triangle so each undirected pair appears once. ROC-AUC and average precision use continuous scores where both classes exist.

The robustness design changes one primary factor at a time. Noise standard deviation, propagation strength and shock count each have four settings. Ten independent development/test seed pairs are used per setting in the full run. Reports include mean and sample standard deviation.

For real prices, dates are aligned, short missing gaps are forward-filled, log returns are computed and each company is regressed on the chosen market index. Residuals become abnormal returns and are standardized by company. Events require \(|z|\geq2.5\), a five-observation cooldown and enough future observations for a complete response window.

## 6 Implementation

Python modules use NumPy and pandas for numerical work, scikit-learn for ranking metrics, NetworkX for network representations, Matplotlib for saved scientific figures and Plotly plus Streamlit for interaction. All random behavior uses explicit NumPy generators. Invalid shapes, unstable coefficients, non-finite values, impossible shock schedules and missing data produce explicit errors.

The dashboard contains overview, synthetic experiment, shock detection, inferred network, method comparison, real-data analysis, and methodology/limitations pages. It can load the packaged cache without internet access and permits a user-supplied adjusted-price CSV.

## 7 Experimental Setup and Results

The default synthetic experiment uses ten companies, 260 time steps, edge probability 0.22, persistence 0.20, propagation 0.55, noise standard deviation 0.10, shock size 2.5 and 30 events. Development seeds are distinct from test seeds. The results below were generated from the recorded configuration and remain linked to raw CSV files under `reports/results/`.

### 7.1 Baseline comparison

Across ten independent test seeds, SRCS achieved mean F1 0.951 (SD 0.048), followed by maximum lagged correlation at 0.921 (SD 0.070), Granger-style predictive gain at 0.917 (SD 0.051), Pearson correlation at 0.623 (SD 0.050), and random ranking at 0.349 (SD 0.096). Mean SRCS ROC-AUC was 0.999 and mean average precision was 0.998. These results apply only to the configured synthetic generator.

The comparison must be reported without assuming SRCS wins. ROC-AUC evaluates ranking across false-positive rates, while average precision is informative for sparse positive edges. Thresholded F1 measures the calibrated classification decision.

### 7.2 Robustness

Noise reduced SRCS performance at the highest tested setting: mean F1 fell from 0.953 at noise 0.05 to 0.777 at noise 0.35. Stronger propagation improved recoverability from mean F1 0.649 at strength 0.20 to 0.978 at 0.55; performance at 0.70 remained high at 0.961. Increasing events from 10 to 30 improved mean F1 from 0.883 to 0.978, while 40 events produced 0.944. Every setting used ten independent development seeds and ten independent test seeds.

Interpretation should distinguish a change in mean performance from variability across seeds. Small networks can show considerable metric variance because a few edge decisions materially change precision or recall.

### 7.3 Real-data demonstration

The verified cache contains 1,003 adjusted daily closing-price observations from 3 January 2022 through 31 December 2025. It covers ten US-listed companies and SPY, with Yahoo Finance via `yfinance` recorded as the source. The detector found 196 events. GOOGL-AMZN ranked first with SRCS 0.2352 and AAPL-MSFT ranked second with 0.2232. GOOGL-NVDA ranked last with 0.0192 and provides an ambiguous low-response case. These rankings may reflect shared sectors or news and do not establish direct economic links.

Two high-ranked cases and one lower or ambiguous case should be discussed only when the cached observations support them. An ambiguous case is scientifically useful because a high market response can arise from shared sector exposure or news rather than a direct economic relationship.

## 8 Testing and Deployment

Unit tests cover simulator reproducibility and validation, zero-variance standardization, event windows, thresholding, single-count metrics, every baseline, price preprocessing, shock cooldowns, cache provenance and an independent calibration-to-inference integration path. GitHub Actions installs the declared dependencies, executes the tests, runs quick experiment variants and starts Streamlit headlessly.

The Docker image uses Python 3.12 and exposes port 8501. Streamlit Community Cloud deployment requires selecting this repository and `app.py`; no runtime secret is required for cached data. External data refresh remains a separate online action.

## 9 Limitations Ethics and Risks

The synthetic data-generating process is linear and static. Real economic connections may be directed, time-varying and nonlinear. Price responses occur at daily resolution and can overlap. Market residualization removes only a linear common component. Extreme observations may reflect earnings, macroeconomic news, data errors or volatility changes. Multiple candidate pairs increase the chance of spurious rankings. Granger-style prediction does not establish structural causality.

Ethically, the dashboard must avoid presenting inferred links as facts about companies. Users should retain provenance, dates and uncertainty and should validate candidates against independent public evidence. The system is an educational research prototype, not investment advice.

## 10 Conclusion and Future Work

Track A establishes a reproducible way to test event-conditioned reconstruction under known ground truth and to compare it with transparent alternatives. The real-data extension demonstrates how the same score can organize event evidence while preserving a strict interpretation boundary. Future work may investigate direction, changing networks, sector controls, bootstrap uncertainty and formal multiple-testing corrections. These extensions belong outside Track A and should follow validation of the current benchmark.

## References

1. Anand, K., van Lelyveld, I., Banai, Á., Friedrich, S., Garratt, R., Hałaj, G., Fique, J., Hansen, I., Jaramillo, S. M., Lee, H., Molina-Borboa, J. L., Nobili, S., Rajan, S., Salakhova, D., Silva, T. C., Silvestri, L., and de Souza, S. R. S. (2018). Reconstruction methods for networks: The case of economic and financial systems. *Physics Reports*, 757, 1–47. https://doi.org/10.1016/j.physrep.2018.06.008
2. Newman, M. E. J. (2010). *Networks: An Introduction*. Oxford University Press. https://doi.org/10.1093/acprof:oso/9780199206650.001.0001
3. MacKinlay, A. C. (1997). Event studies in economics and finance. *Journal of Economic Literature*, 35(1), 13–39. https://doi.org/10.1257/jel.35.1.13
4. Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424–438. https://doi.org/10.2307/1912791

## Appendices

### Appendix A Reproduction commands

See `README.md` and `reports/DEMO_SCRIPT.md`.

### Appendix B Institutional placeholders

- Certificate text and signatures
- Student roll number and division
- Guide and head-of-department approval
- College logo and mandated page layout
- Similarity report and submission date
