"""Streamlit dashboard for the Track A prototype."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.baselines import all_method_scores
from src.data_loader import load_price_csv
from src.evaluation import calibrate_threshold, precision_recall_f1, probability_log_loss
from src.ml_inference import generate_diverse_corpus, predict_edge_scores, train_ml_network_model
from src.network_builder import threshold_scores
from src.preprocessing import preprocess_prices
from src.response_score import build_srcs_matrix, event_evidence_table
from src.shock_detector import detect_shocks, to_srcs_events
from src.synthetic_generator import SyntheticConfig, simulate

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Hidden Economic Connections", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def synthetic_run(config_dict: dict) -> tuple:
    config = SyntheticConfig(**config_dict)
    data = simulate(config)
    development = [simulate(replace(config, seed=seed)) for seed in (101, 102, 103)]
    development_scores = [build_srcs_matrix(item.states, item.events) for item in development]
    threshold = calibrate_threshold(development_scores, [item.adjacency for item in development])
    scores = build_srcs_matrix(data.states, data.events)
    inferred = threshold_scores(scores, threshold)
    return data, scores, inferred, threshold


@st.cache_resource(show_spinner="Training the ML edge classifier on diverse synthetic networks…")
def dashboard_ml_model():
    """Train once per app process using disjoint domain-randomized seed groups."""
    training = generate_diverse_corpus(range(0, 24))
    calibration = generate_diverse_corpus(range(100, 108))
    return train_ml_network_model(training, calibration)[0]


def network_figure(adjacency: np.ndarray, title: str) -> go.Figure:
    graph = nx.from_numpy_array(adjacency)
    positions = nx.spring_layout(graph, seed=7)
    edges_x, edges_y = [], []
    for left, right in graph.edges:
        edges_x += [positions[left][0], positions[right][0], None]
        edges_y += [positions[left][1], positions[right][1], None]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=edges_x, y=edges_y, mode="lines", line={"color": "#94a3b8"}, hoverinfo="skip"))
    figure.add_trace(go.Scatter(x=[positions[n][0] for n in graph.nodes], y=[positions[n][1] for n in graph.nodes],
                                mode="markers+text", text=[f"C{n}" for n in graph.nodes], textposition="top center",
                                marker={"size": 20, "color": "#2563eb"}, hovertemplate="Company %{text}<extra></extra>"))
    figure.update_layout(title=title, showlegend=False, height=430, margin=dict(l=10, r=10, t=45, b=10),
                         xaxis={"visible": False}, yaxis={"visible": False})
    return figure


st.sidebar.title("Track A prototype")
page = st.sidebar.radio("Navigate", ["Project Overview", "Synthetic Experiment", "Shock Detection",
    "Inferred Network", "Method Comparison", "Real-Data Analysis", "Methodology and Limitations"])

with st.sidebar.expander("Synthetic parameters", expanded=False):
    companies = st.slider("Companies", 5, 20, 10)
    noise = st.slider("Noise", 0.01, 0.40, 0.10, 0.01)
    propagation = st.slider("Propagation", 0.10, 0.70, 0.55, 0.05)
    shocks = st.slider("Shocks", 5, 40, 30)
    seed = st.number_input("Test seed", 0, 100000, 42)
config = SyntheticConfig(companies=companies, noise_std=noise, propagation=propagation,
                         shocks=shocks, seed=int(seed))
if config.persistence + config.propagation >= 1:
    st.error("Persistence plus propagation must remain below 1.")
    st.stop()
data, scores, inferred, calibrated_threshold = synthetic_run(config.__dict__)

if page == "Project Overview":
    st.title("Inferring Hidden Economic Connections")
    st.subheader("Synthetic and real-data Track A prototype")
    st.write("This project tests whether repeated, delayed market responses after company-specific shocks contain useful evidence about a hidden static network.")
    st.info("Scores are statistical evidence for candidate connections. They do not prove causality, supplier relationships, contracts or direction of influence.")
    st.markdown("**Workflow:** simulate or load prices → remove common market movement → detect shocks → score delayed responses → rank or threshold candidate edges → evaluate only where ground truth exists.")

elif page == "Synthetic Experiment":
    st.title("Synthetic Experiment")
    metrics = precision_recall_f1(data.adjacency, inferred)
    left, middle, right, accuracy = st.columns(4)
    left.metric("Precision", f"{metrics['precision']:.3f}")
    middle.metric("Recall", f"{metrics['recall']:.3f}")
    right.metric("F1", f"{metrics['f1']:.3f}")
    accuracy.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    st.caption(f"Threshold {calibrated_threshold:.4f}, calibrated only on seeds 101–103; displayed network uses independent seed {seed}.")
    frame = pd.DataFrame(data.states, columns=[f"C{i}" for i in range(companies)])
    st.plotly_chart(px.line(frame, labels={"index": "Time", "value": "State", "variable": "Company"}), use_container_width=True)
    st.download_button("Download observations", frame.to_csv(index_label="time"), "synthetic_observations.csv")

elif page == "Shock Detection":
    st.title("Synthetic Shock Events")
    events = pd.DataFrame(data.events, columns=["time", "company", "signed_shock"])
    st.dataframe(events, use_container_width=True)
    selected = st.selectbox("Inspect origin company", list(range(companies)))
    responders = [i for i in range(companies) if i != selected]
    responder = st.selectbox("Responder", responders)
    st.dataframe(pd.DataFrame(event_evidence_table(data.states, data.events, selected, responder)), use_container_width=True)

elif page == "Inferred Network":
    st.title("True and Inferred Networks")
    ml_model = dashboard_ml_model()
    ml_scores, _ = predict_edge_scores(ml_model.estimator, data.states, data.events)
    ml_inferred = threshold_scores(ml_scores, ml_model.threshold)
    left, right = st.columns(2)
    left.plotly_chart(network_figure(data.adjacency, "True network (evaluation only)"), use_container_width=True)
    method = right.radio("Inference method", ["ML (HistGradientBoosting)", "Deterministic SRCS"])
    selected_network = ml_inferred if method.startswith("ML") else inferred
    right.plotly_chart(network_figure(selected_network, f"{method} inferred network"), use_container_width=True)
    sr, ml = st.columns(2)
    sr.plotly_chart(px.imshow(scores, text_auto=".2f", color_continuous_scale="Viridis",
                              title="SRCS feature heatmap"), use_container_width=True)
    ml.plotly_chart(px.imshow(ml_scores, text_auto=".2f", zmin=0, zmax=1,
                              color_continuous_scale="Blues", title="ML edge-probability heatmap"),
                    use_container_width=True)
    st.dataframe(pd.DataFrame([
        {"method": "SRCS", **precision_recall_f1(data.adjacency, inferred),
         "log_loss": np.nan},
        {"method": "ML (HistGradientBoosting)",
         **precision_recall_f1(data.adjacency, ml_inferred),
         "log_loss": probability_log_loss(data.adjacency, ml_scores)},
    ]), use_container_width=True)
    st.caption("The ML model was trained on seeds 0–23 and calibrated on seeds 100–107; the displayed seed is not used in fitting unless you choose one of those reserved seeds.")

elif page == "Method Comparison":
    st.title("Method Comparison")
    result_path = ROOT / "reports" / "results" / "comparison" / "method_comparison_summary.csv"
    if result_path.exists():
        comparison = pd.read_csv(result_path)
        st.dataframe(comparison, use_container_width=True)
        st.plotly_chart(px.bar(comparison, x="method", y="f1_mean", error_y="f1_std",
                               title="F1 across independent test seeds"), use_container_width=True)
    else:
        st.warning("Run `python -m experiments.compare_methods` to generate the repeated comparison.")
        one = all_method_scores(data.states, data.events, data.config.seed)
        st.plotly_chart(px.bar(x=list(one), y=[np.max(v) for v in one.values()],
                               labels={"x": "Method", "y": "Maximum pair score"}), use_container_width=True)

elif page == "Real-Data Analysis":
    st.title("Real-Data Analysis")
    cached = ROOT / "data" / "real" / "us_large_caps_2022_2025.csv"
    upload = st.file_uploader("Upload adjusted-close CSV (date index, company columns and market index)", type="csv")
    try:
        prices = pd.read_csv(upload, index_col=0, parse_dates=True) if upload else (load_price_csv(cached) if cached.exists() else None)
        if prices is None:
            st.info("No cached real dataset is present. Run `python -m experiments.acquire_real_data`, or upload a CSV.")
        else:
            market = st.selectbox("Market index column", list(prices.columns), index=list(prices.columns).index("SPY") if "SPY" in prices else len(prices.columns)-1)
            abnormal, z_returns = preprocess_prices(prices, market)
            real_events = detect_shocks(z_returns)
            real_scores = build_srcs_matrix(z_returns.to_numpy(), to_srcs_events(real_events))
            ml_model = dashboard_ml_model()
            real_ml_scores, _ = predict_edge_scores(
                ml_model.estimator, z_returns.to_numpy(), to_srcs_events(real_events)
            )
            candidates = [{"company_a": z_returns.columns[i], "company_b": z_returns.columns[j],
                           "ml_edge_probability": real_ml_scores[i, j], "srcs_score": real_scores[i, j]}
                          for i in range(len(z_returns.columns)) for j in range(i+1, len(z_returns.columns))]
            ranked = pd.DataFrame(candidates).sort_values("ml_edge_probability", ascending=False)
            st.metric("Detected shocks", len(real_events)); st.dataframe(ranked.head(20), use_container_width=True)
            st.plotly_chart(px.imshow(real_scores, x=z_returns.columns, y=z_returns.columns,
                                      color_continuous_scale="Viridis", title="Candidate-connection SRCS"), use_container_width=True)
            st.download_button("Download ranked candidates", ranked.to_csv(index=False), "ranked_candidates.csv")
            st.caption("These rankings indicate repeated delayed co-movement after detected events, not causal or commercial links.")
    except Exception as exc:
        st.error(f"Real-data analysis could not run: {exc}")

else:
    st.title("Methodology and Limitations")
    st.markdown("""
### Method
SRCS conditions pair scoring on detected source-company shocks. Each post-event response receives exponential lag decay. The average response is multiplied by the fraction of events exceeding a response threshold, then opposite directions are averaged for an undirected score.

### Evaluation discipline
Thresholds use separate development seeds. Precision, recall, F1 and ranking metrics use independent test seeds, and undirected pairs are counted once.

### Limitations
The static linear synthetic model is deliberately simplified. Real returns contain common factors, volatility clustering, simultaneous news and measurement error. Market-model residualization cannot remove every confounder. Pairwise Granger-style improvement is predictive rather than causal. Ranked candidates require independent domain validation and must never be described as proven commercial relationships.
""")
