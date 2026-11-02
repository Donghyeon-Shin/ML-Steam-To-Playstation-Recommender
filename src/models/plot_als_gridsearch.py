"""Create four publication-ready charts from Binary ALS grid-search results.

Install:
    python -m pip install pandas numpy matplotlib seaborn

Run (CSV in the same folder):
    python plot_als_gridsearch.py

Run (CSV in another folder):
    python plot_als_gridsearch.py --csv path/to/als_binary_gridsearch.csv

Outputs: four charts, each as a high-resolution PNG and editable SVG.
All metrics are shown as percentages; no missing experiment is fabricated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save images without opening GUI windows.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


REQUIRED_COLUMNS = [
    "factors",
    "regularization",
    "confidence_alpha",
    "iterations",
    "recall_at_10",
    "ndcg_at_10",
    "hit_at_10",
]
KEY_COLUMNS = ["factors", "regularization", "confidence_alpha", "iterations"]
ORANGE = "#D97743"
BLUE = "#247BA0"
FACTOR_COLORS = {16: "#7A8591", 24: ORANGE, 32: BLUE}


def load_data(path: Path) -> pd.DataFrame:
    # The provided CSV has a UTF-8 BOM; utf-8-sig handles that safely.
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")

    duplicated = df.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        duplicates = df.loc[duplicated, KEY_COLUMNS]
        raise ValueError(f"Duplicate hyperparameter combinations:\n{duplicates}")

    for column in ("recall_at_10", "ndcg_at_10", "hit_at_10"):
        if df[column].isna().any() or not df[column].between(0, 1).all():
            raise ValueError(f"Invalid metric values in {column}")
    return df


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",  # Keep text editable in SVG editors.
        }
    )


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}.png and {name}.svg")


def stage1_broad_search(df: pd.DataFrame, out_dir: Path) -> None:
    """3 heatmaps: Factors x Alpha, split by regularization; Iterations=20."""
    regs = [0.01, 0.1, 1.0]
    factors = [32, 64, 128]
    alphas = [10, 20, 40]
    subset = df.loc[
        df.factors.isin(factors)
        & df.regularization.isin(regs)
        & df.confidence_alpha.isin(alphas)
        & df.iterations.eq(20)
    ]
    assert len(subset) == 27, f"Expected 27 initial-search rows; found {len(subset)}"

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.8), sharey=True)
    for index, (ax, reg) in enumerate(zip(axes, regs)):
        matrix = (
            subset.loc[subset.regularization.eq(reg)]
            .pivot(index="factors", columns="confidence_alpha", values="ndcg_at_10")
            .reindex(index=factors, columns=alphas)
            * 100
        )
        sns.heatmap(
            matrix,
            ax=ax,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            vmin=23,
            vmax=37,
            linewidths=1.5,
            linecolor="white",
            square=True,
            cbar=False,
            annot_kws={"fontsize": 12, "weight": "semibold"},
        )
        ax.set_title(f"Regularization = {reg:g}", pad=12)
        ax.set_xlabel("Confidence alpha")
        ax.set_ylabel("Factors" if index == 0 else "")
        ax.tick_params(axis="both", length=0)
        ax.grid(False)
    fig.subplots_adjust(right=0.89, wspace=0.12)
    colorbar_axis = fig.add_axes([0.92, 0.23, 0.016, 0.54])
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=23, vmax=37), cmap="YlGnBu"),
        cax=colorbar_axis,
    )
    colorbar.set_label("NDCG@10 (%)")
    fig.suptitle("Stage 1 | Broad hyperparameter search", fontsize=17, weight="bold", y=1.09)
    fig.text(0.5, 1.005, "Iterations = 20  |  Cell values: NDCG@10 (%)", ha="center", fontsize=10, color="#596579")
    save(fig, out_dir, "01_broad_search_heatmaps")


def stage2_low_dim_search(df: pd.DataFrame, out_dir: Path) -> None:
    """NDCG lines by factors; uses identical stage-1 rows where applicable."""
    regs = [0.1, 1.0, 5.0]
    factors = [16, 24, 32]
    alphas = [2, 5, 10]
    subset = df.loc[
        df.factors.isin(factors)
        & df.regularization.isin(regs)
        & df.confidence_alpha.isin(alphas)
        & df.iterations.eq(20)
    ]
    assert len(subset) == 27, (
        f"Expected 27 low-alpha rows (including reused stage-1 measurements); found {len(subset)}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.9), sharey=True)
    for index, (ax, reg) in enumerate(zip(axes, regs)):
        for factor in factors:
            series = subset.loc[
                subset.regularization.eq(reg) & subset.factors.eq(factor)
            ].sort_values("confidence_alpha")
            ax.plot(
                series.confidence_alpha,
                series.ndcg_at_10 * 100,
                marker="o",
                linewidth=2.5 if factor == 24 else 1.9,
                markersize=7,
                color=FACTOR_COLORS[factor],
                label=f"Factors = {factor}",
            )
        ax.set_title(f"Regularization = {reg:g}", pad=12)
        ax.set_xticks(alphas)
        ax.set_xlabel("Confidence alpha")
        ax.set_ylabel("NDCG@10 (%)" if index == 0 else "")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    # Apply one shared scale so the three regularization panels are comparable.
    lo, hi = subset.ndcg_at_10.min() * 100, subset.ndcg_at_10.max() * 100
    axes[0].set_ylim(lo - 0.6, hi + 0.65)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=10)
    fig.suptitle("Stage 2 | Lower factors and lower alpha", fontsize=17, weight="bold", y=1.17)
    fig.text(0.5, 1.08, "Iterations = 20  |  Orange: promising Factors = 24", ha="center", fontsize=10, color="#596579")
    save(fig, out_dir, "02_low_alpha_lineplots")


def stage3_fine_search(df: pd.DataFrame, out_dir: Path) -> None:
    """Fine-search heatmap at Reg=1, Iterations=20; missing trials stay blank."""
    factors = [20, 24, 28, 32]
    alphas = [0, 1, 2, 3, 5]
    subset = df.loc[
        df.factors.isin(factors)
        & df.regularization.eq(1.0)
        & df.confidence_alpha.isin(alphas)
        & df.iterations.eq(20)
    ]
    matrix = (
        subset.pivot(index="factors", columns="confidence_alpha", values="ndcg_at_10")
        .reindex(index=factors, columns=alphas)
        * 100
    )

    fig, ax = plt.subplots(figsize=(10.4, 5.5))
    ax.set_facecolor("#F0F2F5")  # Gray cells represent experiments not run.
    sns.heatmap(
        matrix,
        ax=ax,
        mask=matrix.isna(),
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=25,
        vmax=39,
        linewidths=1.6,
        linecolor="white",
        cbar_kws={"label": "NDCG@10 (%)", "shrink": 0.8},
        annot_kws={"fontsize": 11, "weight": "medium"},
    )
    for i, factor in enumerate(factors):
        for j, alpha in enumerate(alphas):
            if pd.isna(matrix.loc[factor, alpha]):
                ax.text(j + 0.5, i + 0.5, "n/a", ha="center", va="center", color="#85909D", fontsize=10)

    # Highlight the (Factors=24, Alpha=2) candidate within this fixed-reg slice.
    ax.add_patch(Rectangle((alphas.index(2), factors.index(24)), 1, 1, fill=False, edgecolor=ORANGE, linewidth=3.2))
    ax.set_xlabel("Confidence alpha")
    ax.set_ylabel("Factors")
    ax.tick_params(length=0)
    ax.grid(False)
    ax.set_title("Regularization = 1.0   |   Iterations = 20", pad=15)
    fig.suptitle("Stage 3 | Fine search (observed combinations only)", fontsize=17, weight="bold", y=1.055)
    fig.text(0.5, -0.025, "Gray 'n/a' = not tested   |   Orange outline = Factors 24, alpha 2 candidate", ha="center", fontsize=10, color="#596579")
    save(fig, out_dir, "03_fine_search_heatmap")


def iteration_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """Recall@10 and NDCG@10 for fixed Factors=24, Reg=5, Alpha=2."""
    subset = df.loc[
        df.factors.eq(24)
        & df.regularization.eq(5.0)
        & df.confidence_alpha.eq(2)
        & df.iterations.isin([10, 20, 40])
    ].sort_values("iterations")
    if subset.iterations.tolist() != [10, 20, 40]:
        raise ValueError("Missing a 10/20/40 iteration result for Factors=24, Reg=5, Alpha=2")

    fig, axes = plt.subplots(1, 2, figsize=(12.1, 4.7))
    for ax, (column, label, color) in zip(
        axes,
        [
            ("recall_at_10", "Recall@10", BLUE),
            ("ndcg_at_10", "NDCG@10", ORANGE),
        ],
    ):
        values = subset[column].to_numpy() * 100
        xs = np.arange(3)
        ax.plot(xs, values, color=color, linewidth=2.8, marker="o", markersize=9)
        ax.scatter([xs[-1]], [values[-1]], s=220, facecolors="none", edgecolors=color, linewidths=1.8, zorder=4)
        spread = max(values.max() - values.min(), 0.12)
        ax.set_ylim(values.min() - spread * 0.65, values.max() + spread * 0.85)
        ax.set_xlim(-0.35, 2.25)
        ax.set_xticks(xs, ["10", "20", "40"])
        ax.set_xlabel("Iterations")
        ax.set_ylabel(f"{label} (%)")
        ax.set_title(label, pad=12)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        for x, value in zip(xs, values):
            ax.annotate(f"{value:.2f}%", (x, value), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=11, weight="bold" if x == 2 else "normal")

    fig.suptitle("Iterations comparison | Fixed hyperparameters", fontsize=17, weight="bold", y=1.12)
    fig.text(0.5, 1.02, "Factors = 24   |   Regularization = 5.0   |   Confidence alpha = 2", ha="center", fontsize=10, color="#596579")
    fig.text(0.5, -0.035, "Note: Y-axes are zoomed to make small differences visible; values are absolute percentages.", ha="center", fontsize=10, color="#596579")
    save(fig, out_dir, "04_iterations_recall_ndcg")

    print("\nIterations comparison (actual CSV values):")
    print(subset[["iterations", "recall_at_10", "ndcg_at_10"]].to_string(index=False, float_format=lambda x: f"{x * 100:.2f}%"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the three Binary ALS tuning stages and iteration comparison")
    parser.add_argument("--csv", type=Path, default=Path("als_binary_gridsearch.csv"), help="Input grid-search CSV path")
    parser.add_argument("--output-dir", type=Path, default=Path("als_gridsearch_figures"), help="Output image folder")
    args = parser.parse_args()

    if not args.csv.is_file():
        parser.error(f"CSV file not found: {args.csv.resolve()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(args.csv)
    set_style()

    stage1_broad_search(df, args.output_dir)
    stage2_low_dim_search(df, args.output_dir)
    stage3_fine_search(df, args.output_dir)
    iteration_comparison(df, args.output_dir)
    print(f"\nDone. Charts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
