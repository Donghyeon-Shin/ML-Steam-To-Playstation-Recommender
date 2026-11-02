"""Visualize Playtime Weighted ALS grid-search experiments.

Requirements:
    python -m pip install pandas numpy matplotlib seaborn

Run from your repository root:
    python src/models/plot_als_playtime_gridsearch.py --csv data/results/als_playtime_gridsearch.csv --output-dir data/results/als_playtime_figures

The script writes two figures as 300-DPI PNG and editable SVG files.
Metrics are read from the original CSV, with no interpolation or fabricated rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle


REQUIRED_COLUMNS = [
    "factors", "regularization", "iterations", "owner_alpha",
    "playtime_beta", "cap_hours", "recall_at_10", "ndcg_at_10", "hit_at_10",
]
RUN_KEYS = ["factors", "regularization", "iterations", "owner_alpha", "playtime_beta", "cap_hours"]
BETAS_STAGE1 = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
BETAS_STAGE2 = [0.1, 0.25, 0.5, 1.0]
CAP_PERCENTILES = [50, 75, 90, 95, 99]
BLUE = "#2B789E"
ORANGE = "#D8753D"
GREEN = "#3F9380"


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV columns missing: {sorted(missing)}")
    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")
    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("CSV includes missing numeric values")
    if df.duplicated(RUN_KEYS).any():
        raise ValueError("Duplicate hyperparameter combinations exist in the CSV")
    for metric in ("recall_at_10", "ndcg_at_10", "hit_at_10"):
        if not df[metric].between(0, 1).all():
            raise ValueError(f"Invalid values in {metric}")
    return df


def fixed_model(df: pd.DataFrame) -> pd.DataFrame:
    """Both requested experiments fix Factors=24, Reg=5, Iterations=40, owner alpha=2."""
    subset = df.loc[
        df.factors.eq(24)
        & df.regularization.eq(5.0)
        & df.iterations.eq(40)
        & df.owner_alpha.eq(2)
    ].copy()
    if subset.empty:
        raise ValueError("No results for Factors=24, Reg=5.0, Iterations=40, Owner alpha=2")
    return subset


def style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, output: Path, filename: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(output / f"{filename}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}.png / {filename}.svg")


def stage1_beta_sensitivity(fixed: pd.DataFrame, output: Path) -> tuple[float, float]:
    """At the 99th percentile cap, compare all seven beta settings."""
    cap99 = fixed.cap_hours.max()
    subset = fixed.loc[np.isclose(fixed.cap_hours, cap99)].sort_values("playtime_beta")
    if not np.allclose(subset.playtime_beta.to_numpy(), BETAS_STAGE1):
        raise ValueError("Stage 1 needs beta = 0, 0.1, 0.25, 0.5, 1, 2, 4 at cap99")

    specs = [
        ("recall_at_10", "Recall@10", BLUE),
        ("ndcg_at_10", "NDCG@10", ORANGE),
        ("hit_at_10", "Hit Rate@10", GREEN),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    xs = np.arange(len(BETAS_STAGE1))
    for ax, (metric, label, color) in zip(axes, specs):
        scores = subset[metric].to_numpy() * 100
        ax.plot(xs, scores, color=color, linewidth=2.5, marker="o", markersize=7)
        ax.scatter([xs[0]], [scores[0]], s=200, facecolors="none", edgecolors=color, linewidths=1.7, zorder=4)
        ax.set_xticks(xs, [f"{beta:g}" for beta in BETAS_STAGE1])
        ax.set_xlim(-0.35, 6.35)
        ax.set_xlabel("Playtime beta (tested values)", fontsize=10)
        ax.set_ylabel(f"{label} (%)", fontsize=10)
        ax.set_title(label, pad=12)
        gap = max(scores.max() - scores.min(), 0.07)
        ax.set_ylim(scores.min() - 0.25 * gap, scores.max() + 0.3 * gap)
        ax.grid(alpha=0.23)
        ax.spines[["top", "right"]].set_visible(False)
        ax.annotate(f"{scores[0]:.2f}%", (0, scores[0]), xytext=(8, 13), textcoords="offset points", fontsize=9, weight="bold")
        ax.annotate(f"{scores[-1]:.2f}%", (6, scores[-1]), xytext=(0, -20), textcoords="offset points", ha="center", fontsize=9)

    fig.suptitle("Stage 1 | Beta sensitivity at the 99th-percentile cap", fontsize=17, weight="bold", y=1.12)
    fig.text(0.5, 1.025, f"Cap = {cap99:.2f} hours   |   Factors = 24, Reg = 5, Iterations = 40, Owner alpha = 2", ha="center", fontsize=10, color="#596579")
    fig.text(0.5, -0.045, "Tested beta settings are equally spaced on the x-axis; y-axes are zoomed to show score differences.", ha="center", fontsize=9, color="#596579")
    save_figure(fig, output, "01_high_cap_beta_sensitivity")
    binary_row = subset.loc[np.isclose(subset.playtime_beta, 0.0)].iloc[0]
    return float(binary_row.ndcg_at_10), float(binary_row.recall_at_10)


def stage2_matrix(fixed: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    # The second search uses four beta settings across five positive-playtime cap percentiles.
    caps = sorted(fixed.cap_hours.unique())
    if len(caps) != 5:
        raise ValueError(f"Expected exactly 5 cap values; found {len(caps)}: {caps}")
    subset = fixed.loc[
        fixed.playtime_beta.isin(BETAS_STAGE2)
        & fixed.cap_hours.isin(caps)
    ].copy()
    if len(subset) != 20:
        raise ValueError(f"Expected 20 cap/beta results; found {len(subset)}")
    return subset, caps


def heatmaps_by_cap_and_beta(subset: pd.DataFrame, caps: list[float], output: Path) -> None:
    """NDCG + Recall heatmaps; highlight best observed cell for each metric."""
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2))
    row_labels = [f"P{p}  |  {cap:.2f} h" for p, cap in zip(CAP_PERCENTILES, caps)]
    beta_labels = [f"{b:g}" for b in BETAS_STAGE2]

    for ax, (metric, title) in zip(
        axes, [("ndcg_at_10", "NDCG@10 (%)"), ("recall_at_10", "Recall@10 (%)")]
    ):
        grid = (
            subset.pivot(index="cap_hours", columns="playtime_beta", values=metric)
            .reindex(index=caps, columns=BETAS_STAGE2)
            * 100
        )
        if grid.isna().any().any():
            raise ValueError(f"Missing Stage-2 combination for {metric}")

        sns.heatmap(
            grid,
            ax=ax,
            cmap="YlGnBu",
            annot=True,
            fmt=".2f",
            linewidths=1.3,
            linecolor="white",
            cbar=metric == "recall_at_10",
            cbar_kws={"label": title, "shrink": 0.8} if metric == "recall_at_10" else None,
            annot_kws={"fontsize": 10},
        )
        ax.set_xticklabels(beta_labels, rotation=0)
        if metric == "ndcg_at_10":
            ax.set_yticklabels(row_labels, rotation=0)
        else:
            ax.set_yticklabels([])  # Both heatmaps use the same row ordering.
            ax.tick_params(axis="y", left=False)
        ax.set_xlabel("Playtime beta")
        ax.set_ylabel("Cap percentile / cap hours" if metric == "ndcg_at_10" else "")
        ax.set_title(title, pad=13)
        ax.tick_params(axis="both", length=0)
        ax.grid(False)

        # Outline the best measured cell, which may differ across NDCG and Recall.
        best_index = np.unravel_index(np.argmax(grid.to_numpy()), grid.shape)
        ax.add_patch(Rectangle((best_index[1], best_index[0]), 1, 1, fill=False, edgecolor=ORANGE, linewidth=3.2))

    fig.subplots_adjust(wspace=0.25)
    fig.suptitle("Stage 2 | Cap percentile × Beta search", fontsize=17, weight="bold", y=1.075)
    fig.text(0.5, -0.035, "Orange outline = maximum observed value for that metric. Model hyperparameters are fixed.", ha="center", fontsize=9, color="#596579")
    save_figure(fig, output, "02_cap_beta_heatmaps")


def summarize_results(
    subset: pd.DataFrame,
    baseline_ndcg: float,
    baseline_recall: float,
) -> None:
    """Print best observed settings without generating an additional graph."""
    best = (
        subset.sort_values(["cap_hours", "ndcg_at_10", "playtime_beta"], ascending=[True, False, True])
        .groupby("cap_hours", as_index=False)
        .head(1)
        .sort_values("cap_hours")
    )
    if len(best) != 5:
        raise ValueError("Could not select an NDCG-best beta for each cap")

    print("\nBest beta for each cap (ranked by NDCG@10):")
    printed = best[["cap_hours", "playtime_beta", "recall_at_10", "ndcg_at_10"]].copy()
    printed[["recall_at_10", "ndcg_at_10"]] *= 100
    print(printed.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    top = best.loc[best.ndcg_at_10.idxmax()]
    print(f"\nBest weighted NDCG = {top.ndcg_at_10 * 100:.4f}% (cap={top.cap_hours:.2f} h, beta={top.playtime_beta:g})")
    print(f"Beta=0 baseline NDCG = {baseline_ndcg * 100:.4f}%")
    print(f"Difference = {(top.ndcg_at_10 - baseline_ndcg) * 100:+.4f} percentage points")
    print(f"Beta=0 baseline Recall = {baseline_recall * 100:.4f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Playtime Weighted ALS beta and cap experiments")
    parser.add_argument("--csv", type=Path, default=Path("als_playtime_gridsearch.csv"), help="Input CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("als_playtime_figures"), help="Output directory")
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"CSV does not exist: {args.csv.resolve()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.csv)
    fixed = fixed_model(df)
    style()
    baseline_ndcg, baseline_recall = stage1_beta_sensitivity(fixed, args.output_dir)
    subset, caps = stage2_matrix(fixed)
    heatmaps_by_cap_and_beta(subset, caps, args.output_dir)
    summarize_results(subset, baseline_ndcg, baseline_recall)
    print(f"\nDone. Files saved under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
