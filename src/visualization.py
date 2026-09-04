"""Figures needed to verify the Step 1 synthetic environment."""

from pathlib import Path

import matplotlib

# Figures are generated in terminals and CI too, where a GUI/Tk installation is absent.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from src.synthetic_generator import SyntheticData


def save_figures(data: SyntheticData, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _architecture(output_dir / "system_architecture.png")
    _true_network(data, output_dir / "true_network.png")
    _propagation(data, output_dir / "shock_propagation.png")


def _architecture(path: Path) -> None:
    labels = ["Hidden network", "Shocks + noise", "Company states", "Later inference"]
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


def _true_network(data: SyntheticData, path: Path) -> None:
    graph = nx.from_numpy_array(data.adjacency)
    layout = nx.spring_layout(graph, seed=7)
    figure, axis = plt.subplots(figsize=(6, 5))
    nx.draw_networkx(graph, layout, ax=axis, node_color="#6aaed6", edge_color="#4a5568",
                     node_size=700, font_weight="bold")
    axis.set_title("Step 1: true hidden company network")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


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
