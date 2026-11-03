
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# 1. Path configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULT_DIR = PROJECT_DIR / "data" / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Load playtime data
# ============================================================

def load_playtime():

    df = pd.read_parquet(
        DATA_DIR / "steam_train.parquet",
        columns=["playtime_forever"]
    )

    playtime = pd.to_numeric(
        df["playtime_forever"],
        errors="coerce"
    )

    print("=" * 55)
    print("Steam Playtime Distribution Analysis")
    print("=" * 55)

    print(f"전체 소유 기록: {len(df):,}건")
    print(f"결측값: {playtime.isna().sum():,}건")
    print(f"음수: {(playtime < 0).sum():,}건")

    # Exclude missing and invalid playtime values.
    playtime = playtime[
        playtime.notna() & (playtime >= 0)
    ]

    return playtime


# ============================================================
# 3. Descriptive statistics
# ============================================================

def analyze_statistics(playtime):

    total = len(playtime)

    zero_count = (playtime == 0).sum()
    positive = playtime[playtime > 0]

    print("\n" + "=" * 55)
    print("1. 기본 통계")
    print("=" * 55)

    print(f"유효한 소유 기록: {total:,}건")

    print(
        f"플레이타임 0분: {zero_count:,}건 "
        f"({zero_count / total * 100:.2f}%)"
    )

    print(
        f"플레이타임 0분 초과: {len(positive):,}건 "
        f"({len(positive) / total * 100:.2f}%)"
    )

    print("\n전체 소유 게임 기준")

    print(f"평균: {playtime.mean() / 60:.2f}시간")
    print(f"중앙값: {playtime.median() / 60:.2f}시간")
    print(f"최댓값: {playtime.max() / 60:.2f}시간")

    print("\n실제 플레이한 게임 기준 (0분 제외)")

    if positive.empty:
        print("플레이타임이 0분을 초과하는 기록이 없습니다.")
        return

    print(f"평균: {positive.mean() / 60:.2f}시간")
    print(f"중앙값: {positive.median() / 60:.2f}시간")
    print(f"최댓값: {positive.max() / 60:.2f}시간")

    # Calculate quantiles for games with positive playtime.
    percentiles = [
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        0.999
    ]

    quantiles = positive.quantile(percentiles)

    print("\n플레이타임 분위수 (0분 제외)")

    for percentile, value in quantiles.items():

        print(
            f"{percentile * 100:5.1f}%: "
            f"{value / 60:,.2f}시간"
        )


# ============================================================
# 4. Playtime distribution by interval
# ============================================================

def analyze_intervals(playtime):

    print("\n" + "=" * 55)
    print("2. 플레이타임 구간별 분포")
    print("=" * 55)

    # Define interval boundaries in minutes.
    bins = [
        -np.inf,
        0,
        30,
        120,
        600,
        1800,
        6000,
        np.inf
    ]

    labels = [
        "0분",
        "0~30분",
        "30분~2시간",
        "2~10시간",
        "10~30시간",
        "30~100시간",
        "100시간 초과"
    ]

    intervals = pd.cut(
        playtime,
        bins=bins,
        labels=labels
    )

    counts = intervals.value_counts(sort=False)

    percentages = counts / len(playtime) * 100

    for label in labels:

        count = counts[label]
        percentage = percentages[label]

        print(
            f"{label:14s}: "
            f"{count:10,}건 "
            f"({percentage:6.2f}%)"
        )

    return percentages


# ============================================================
# 5. Plot the playtime interval distribution
# ============================================================

def plot_intervals(percentages):

    labels = [
        "0 min",
        "0-30 min",
        "0.5-2 h",
        "2-10 h",
        "10-30 h",
        "30-100 h",
        "100+ h"
    ]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(
        labels,
        percentages.values,
        color="steelblue"
    )

    ax.set_title(
        "Steam Playtime Distribution",
        fontsize=15
    )

    ax.set_xlabel("Playtime")
    ax.set_ylabel("Percentage of Owned Games (%)")

    ax.tick_params(axis="x", rotation=25)

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = RESULT_DIR / "playtime_intervals.png"

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(f"\n구간별 그래프 저장: {output_path}")


# ============================================================
# 6. Plot the log-scale playtime distribution
# ============================================================

def plot_log_distribution(playtime):

    # Exclude games with zero playtime.
    positive = playtime[playtime > 0]

    if positive.empty:
        return

    # Convert minutes to hours.
    playtime_hours = positive.to_numpy() / 60

    # Apply a logarithmic transformation.
    log_playtime = np.log1p(playtime_hours)

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.hist(
        log_playtime,
        bins=100,
        color="darkorange",
        edgecolor="white"
    )

    # Label the x-axis using actual playtime in hours.
    hour_ticks = [
        0,
        0.5,
        2,
        10,
        50,
        200,
        1000
    ]

    ax.set_xticks(
        np.log1p(hour_ticks)
    )

    ax.set_xticklabels(
        [str(x) for x in hour_ticks]
    )

    ax.set_title(
        "Steam Playtime Distribution (Log Scale)",
        fontsize=15
    )

    ax.set_xlabel("Playtime (Hours, Log Scale)")
    ax.set_ylabel("Number of Owned Games")

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = RESULT_DIR / "playtime_log_distribution.png"

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(f"로그 스케일 그래프 저장: {output_path}")


# ============================================================
# 7. Main execution
# ============================================================

def main():

    # Load the playtime data.
    playtime = load_playtime()

    if playtime.empty:
        print("분석 가능한 데이터가 없습니다.")
        return

    # Print descriptive statistics.
    analyze_statistics(playtime)

    # Calculate the interval distribution.
    percentages = analyze_intervals(playtime)

    # Generate distribution plots.
    plot_intervals(percentages)

    plot_log_distribution(playtime)

    print("\n플레이타임 분포 분석 완료")


if __name__ == "__main__":
    main()
