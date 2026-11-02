
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. 경로 및 설정
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULT_DIR = PROJECT_DIR / "data" / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

CONCENTRATION_THRESHOLD = 0.8


# ============================================================
# 2. 데이터 불러오기
# ============================================================

def load_data():

    df = pd.read_parquet(
        DATA_DIR / "steam_train.parquet",
        columns=[
            "user_id",
            "unified_game_id",
            "playtime_forever"
        ]
    )

    df["playtime_forever"] = pd.to_numeric(
        df["playtime_forever"],
        errors="coerce"
    )

    # 데이터 검증
    if df["playtime_forever"].isna().any():
        raise ValueError("플레이타임 결측값이 존재합니다.")

    if (df["playtime_forever"] < 0).any():
        raise ValueError("음수 플레이타임이 존재합니다.")

    if df.duplicated(
        ["user_id", "unified_game_id"]
    ).any():
        raise ValueError("사용자-게임 중복 기록이 존재합니다.")

    print("=" * 60)
    print("Steam User Playtime Concentration Analysis")
    print("=" * 60)

    print(f"전체 소유 기록: {len(df):,}건")
    print(f"전체 사용자: {df['user_id'].nunique():,}명")

    return df


# ============================================================
# 3. 사용자별 플레이타임 통계
# ============================================================

def calculate_user_statistics(df):
    """
    사용자별로 다음 항목을 계산합니다.

    owned_games:
        소유한 게임 수

    played_games:
        플레이타임이 0보다 큰 게임 수

    total_minutes:
        전체 누적 플레이타임

    max_minutes:
        가장 많이 플레이한 게임의 플레이타임

    top1_share:
        전체 플레이타임 중 최다 플레이 게임의 비중
    """

    df = df.copy()

    df["played_flag"] = (
        df["playtime_forever"] > 0
    ).astype(np.int8)

    user_stats = (
        df.groupby("user_id")
        .agg(
            owned_games=(
                "unified_game_id",
                "size"
            ),
            played_games=(
                "played_flag",
                "sum"
            ),
            total_minutes=(
                "playtime_forever",
                "sum"
            ),
            max_minutes=(
                "playtime_forever",
                "max"
            )
        )
        .reset_index()
    )

    # 시간 단위로 변환
    user_stats["total_hours"] = (
        user_stats["total_minutes"] / 60
    )

    user_stats["max_hours"] = (
        user_stats["max_minutes"] / 60
    )

    # Top-1 플레이타임 집중도
    user_stats["top1_share"] = np.where(
        user_stats["total_minutes"] > 0,
        (
            user_stats["max_minutes"]
            / user_stats["total_minutes"]
        ),
        np.nan
    )

    return user_stats


# ============================================================
# 4. 전체 사용자 집중도 분석
# ============================================================

def analyze_overall(user_stats):

    print("\n" + "=" * 60)
    print("1. 전체 사용자 집중도")
    print("=" * 60)

    total_users = len(user_stats)

    no_play = user_stats[
        user_stats["played_games"] == 0
    ]

    one_game = user_stats[
        user_stats["played_games"] == 1
    ]

    multi_game = user_stats[
        user_stats["played_games"] >= 2
    ]

    print(f"전체 사용자: {total_users:,}명")

    print(
        f"플레이한 게임 없음: {len(no_play):,}명 "
        f"({len(no_play) / total_users * 100:.2f}%)"
    )

    print(
        f"실제 플레이한 게임 1개: {len(one_game):,}명 "
        f"({len(one_game) / total_users * 100:.2f}%)"
    )

    print(
        f"실제 플레이한 게임 2개 이상: {len(multi_game):,}명 "
        f"({len(multi_game) / total_users * 100:.2f}%)"
    )

    # 실제 플레이한 게임이 2개 이상인 사용자 분석
    print("\n실제 플레이한 게임 2개 이상 사용자 기준")

    if multi_game.empty:
        print("분석 대상 사용자가 없습니다.")
        return

    print(
        f"평균 집중도: "
        f"{multi_game['top1_share'].mean() * 100:.2f}%"
    )

    print(
        f"중앙값 집중도: "
        f"{multi_game['top1_share'].median() * 100:.2f}%"
    )

    for threshold in [0.5, 0.8, 0.9]:

        count = (
            multi_game["top1_share"] >= threshold
        ).sum()

        percentage = (
            count / len(multi_game) * 100
        )

        print(
            f"집중도 {threshold * 100:.0f}% 이상: "
            f"{count:,}명 ({percentage:.2f}%)"
        )


# ============================================================
# 5. 장시간 플레이 사용자 집중도 분석
# ============================================================

def analyze_heavy_players(user_stats):
    """
    최다 플레이 게임이 특정 시간 이상인 사용자들을
    분석합니다.
    """

    print("\n" + "=" * 60)
    print("2. 장시간 플레이 사용자 집중도")
    print("=" * 60)

    thresholds = [
        100,
        500,
        1000,
        3000
    ]

    for threshold in thresholds:

        selected = user_stats[
            user_stats["max_hours"] >= threshold
        ]

        print("\n" + "-" * 55)
        print(f"최다 플레이 게임 {threshold:,}시간 이상")
        print("-" * 55)

        if selected.empty:
            print("해당 사용자가 없습니다.")
            continue

        print(f"전체 사용자: {len(selected):,}명")

        # 실제 플레이한 게임이 하나뿐인 사용자
        one_game = selected[
            selected["played_games"] == 1
        ]

        print(
            f"1개 게임만 플레이: {len(one_game):,}명 "
            f"({len(one_game) / len(selected) * 100:.2f}%)"
        )

        # 실제 플레이한 게임이 2개 이상인 사용자
        multi_game = selected[
            selected["played_games"] >= 2
        ]

        if multi_game.empty:
            print("2개 이상 게임을 플레이한 사용자가 없습니다.")
            continue

        print(
            f"2개 이상 게임 플레이: {len(multi_game):,}명"
        )

        print(
            f"집중도 중앙값: "
            f"{multi_game['top1_share'].median() * 100:.2f}%"
        )

        for share_threshold in [0.8, 0.9]:

            count = (
                multi_game["top1_share"]
                >= share_threshold
            ).sum()

            percentage = (
                count / len(multi_game) * 100
            )

            print(
                f"집중도 {share_threshold * 100:.0f}% 이상: "
                f"{count:,}명 ({percentage:.2f}%)"
            )

        print(
            f"실제 플레이한 게임 수 중앙값: "
            f"{multi_game['played_games'].median():.1f}개"
        )


# ============================================================
# 6. 최다 플레이타임 구간별 집중도
# ============================================================

def analyze_by_max_playtime(user_stats):

    print("\n" + "=" * 60)
    print("3. 최다 플레이타임 구간별 집중도")
    print("=" * 60)

    # 1개 게임만 플레이한 사용자는 제외
    multi_game = user_stats[
        user_stats["played_games"] >= 2
    ].copy()

    bins = [
        0,
        2,
        10,
        30,
        100,
        500,
        1000,
        np.inf
    ]

    labels = [
        "0-2h",
        "2-10h",
        "10-30h",
        "30-100h",
        "100-500h",
        "500-1000h",
        "1000h+"
    ]

    multi_game["max_time_bin"] = pd.cut(
        multi_game["max_hours"],
        bins=bins,
        labels=labels
    )

    # 집중도 80% 이상 여부
    multi_game["high_concentration"] = (
        multi_game["top1_share"]
        >= CONCENTRATION_THRESHOLD
    )

    # 구간별 통계
    summary = (
        multi_game.groupby(
            "max_time_bin",
            observed=False
        )
        .agg(
            users=(
                "user_id",
                "size"
            ),
            median_top1_share=(
                "top1_share",
                "median"
            ),
            high_concentration_rate=(
                "high_concentration",
                "mean"
            ),
            median_played_games=(
                "played_games",
                "median"
            )
        )
        .reset_index()
    )

    # 백분율 변환
    summary["median_top1_share"] *= 100
    summary["high_concentration_rate"] *= 100

    print(
        "\n구간별 사용자 수 / 집중도 중앙값 / "
        "집중도 80% 이상 비율"
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # 결과 저장
    output_path = (
        RESULT_DIR
        / "playtime_concentration_summary.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n분석 결과 저장: {output_path}")

    # 상관관계 계산
    correlation = multi_game[
        ["max_hours", "top1_share"]
    ].corr(method="spearman").iloc[0, 1]

    print(
        f"\n최대 플레이타임과 집중도의 "
        f"Spearman 상관계수: {correlation:.4f}"
    )

    return summary


# ============================================================
# 7. 사용자 집중도 분포 그래프
# ============================================================

def plot_concentration(user_stats):

    active_users = user_stats[
        user_stats["played_games"] >= 1
    ]

    multi_game = user_stats[
        user_stats["played_games"] >= 2
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5)
    )

    bins = np.linspace(0, 100, 51)

    # 1개 이상 게임 플레이한 사용자
    axes[0].hist(
        active_users["top1_share"] * 100,
        bins=bins,
        color="steelblue"
    )

    axes[0].set_title(
        "Top-1 Share (All Active Users)"
    )

    axes[0].set_xlabel("Top-1 Playtime Share (%)")
    axes[0].set_ylabel("Number of Users")

    # 2개 이상 게임 플레이한 사용자
    axes[1].hist(
        multi_game["top1_share"] * 100,
        bins=bins,
        color="darkorange"
    )

    axes[1].set_title(
        "Top-1 Share (2+ Played Games)"
    )

    axes[1].set_xlabel("Top-1 Playtime Share (%)")
    axes[1].set_ylabel("Number of Users")

    for ax in axes:
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_path = (
        RESULT_DIR
        / "playtime_concentration_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(f"\n집중도 분포 그래프 저장: {output_path}")


# ============================================================
# 8. 최대 플레이타임과 집중도의 관계
# ============================================================

def plot_concentration_by_playtime(summary):

    # 사용자가 존재하는 구간만 표시
    summary = summary[
        summary["users"] > 0
    ]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(
        summary["max_time_bin"].astype(str),
        summary["high_concentration_rate"],
        color="seagreen"
    )

    ax.set_title(
        "High Concentration Rate by Maximum Playtime"
    )

    ax.set_xlabel("Maximum Game Playtime")
    ax.set_ylabel("Users with Top-1 Share >= 80% (%)")

    ax.tick_params(axis="x", rotation=25)

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_path = (
        RESULT_DIR
        / "concentration_by_max_playtime.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(f"구간별 집중도 그래프 저장: {output_path}")


# ============================================================
# 9. 메인 실행
# ============================================================

def main():

    # 1. 데이터 로드
    df = load_data()

    # 2. 사용자별 통계 계산
    user_stats = calculate_user_statistics(df)

    # 3. 전체 사용자 집중도
    analyze_overall(user_stats)

    # 4. 장시간 플레이 사용자 분석
    analyze_heavy_players(user_stats)

    # 5. 플레이타임 구간별 집중도
    summary = analyze_by_max_playtime(user_stats)

    # 6. 그래프 생성
    plot_concentration(user_stats)

    plot_concentration_by_playtime(summary)

    print("\n" + "=" * 60)
    print("사용자별 플레이타임 집중도 분석 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
