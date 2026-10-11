"""Figures needed to verify the Step 1 synthetic environment."""

from pathlib import Path

import matplotlib

# Figures are generated in terminals and CI too, where a GUI/Tk installation is absent.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

import numpy as np

from src.synthetic_generator import SyntheticData


def save_figures(data: SyntheticData, output_dir: Path, inferred: np.ndarray | None = None,
                 scores: np.ndarray | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _architecture(output_dir / "system_architecture.png")
    _data_flow(output_dir / "data_flow.png")
    _true_network(data, output_dir / "true_network.png")
    _propagation(data, output_dir / "shock_propagation.png")
    if inferred is not None:
        _network(inferred, output_dir / "inferred_network.png", "Inferred company network")
    if scores is not None:
        _heatmap(scores, output_dir / "srcs_score_heatmap.png")


def _architecture(path: Path) -> None:
    labels = ["Configuration", "Simulation / prices", "Reusable analysis", "Results + dashboard"]
    figure, axis = plt.subplots(figsize=(11, 2.2))
    axis.axis("off")
    for index, label in enumerate(labels):
        x = 0.03 + index * 0.25
        axis.text(x, 0.5, label, ha="center", va="center", fontsize=11,
                  bbox={"boxstyle": "round,pad=0.6", "facecolor": "#e8f0fe", "edgecolor": "#3266a8"})
        if index < len(labels) - 1:
            axis.annotate("", xy=(x + 0.19, 0.5), xytext=(x + 0.09, 0.5),
                          arrowprops={"arrowstyle": "->", "color": "#3266a8", "lw": 1.8})
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _data_flow(path: Path) -> None:
    """Render the shared synthetic and real-data analysis flow."""
    lanes = [
        (0.70, "Synthetic", ["Hidden graph", "Propagated states", "Known edges"]),
        (0.30, "Real data", ["Adjusted prices", "Abnormal returns", "Unknown edges"]),
    ]
    figure, axis = plt.subplots(figsize=(12, 4.2))
    axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.axis("off")
    colors = ["#dceafe", "#e9f7ef", "#fff3cd", "#f5e6ff", "#e8edf3"]
    shared = ["Standardize", "Detect shocks", "Score pairs (SRCS)", "Rank / threshold", "Evidence + figures"]
    for y, lane, inputs in lanes:
        axis.text(0.015, y, lane, ha="left", va="center", weight="bold", color="#23395d")
        axis.text(0.14, y, "\n".join(inputs), ha="center", va="center", fontsize=9,
                  bbox={"boxstyle": "round,pad=0.45", "facecolor": colors[0], "edgecolor": "#3266a8"})
        axis.annotate("", xy=(0.26, y), xytext=(0.21, y),
                      arrowprops={"arrowstyle": "->", "color": "#52606d", "lw": 1.5})
        for index, label in enumerate(shared):
            x = 0.31 + index * 0.16
            axis.text(x, y, label, ha="center", va="center", fontsize=8.5,
                      bbox={"boxstyle": "round,pad=0.38", "facecolor": colors[index], "edgecolor": "#52606d"})
            if index < len(shared) - 1:
                axis.annotate("", xy=(x + 0.12, y), xytext=(x + 0.08, y),
                              arrowprops={"arrowstyle": "->", "color": "#52606d", "lw": 1.4})
    axis.set_title("Track A data flow: shared event-conditioned inference pipeline", pad=16, weight="bold")
    axis.text(0.50, 0.05,
              "Synthetic path supports calibrated edge evaluation; real data produces non-causal candidate rankings.",
              ha="center", va="center", fontsize=9, color="#52606d")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _true_network(data: SyntheticData, path: Path) -> None:
    _network(data.adjacency, path, "True hidden company network")


def _network(adjacency: np.ndarray, path: Path, title: str) -> None:
    graph = nx.from_numpy_array(adjacency)
    layout = nx.spring_layout(graph, seed=7)
    figure, axis = plt.subplots(figsize=(6, 5))
    nx.draw_networkx(graph, layout, ax=axis, node_color="#6aaed6", edge_color="#4a5568",
                     node_size=700, font_weight="bold")
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _heatmap(scores: np.ndarray, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(scores, cmap="viridis")
    axis.set(title="SRCS score matrix", xlabel="Company", ylabel="Company")
    figure.colorbar(image, ax=axis, label="SRCS")
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def _propagation(data: SyntheticData, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    for company in range(data.states.shape[1]):
        axis.plot(data.states[:, company], lw=1.2, label=f"Company {company}")
    for time, _, _ in data.events:
        axis.axvline(time, color="#d1495b", alpha=0.15, lw=1)
    axis.set(title="Step 1: synthetic shock propagation", xlabel="Time step", ylabel="Company state")
    axis.legend(ncol=2, fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
