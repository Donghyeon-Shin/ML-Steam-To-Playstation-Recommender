from pathlib import Path
import math

import pandas as pd


DATA_DIR = Path("data/processed")
TOP_K = 10


def calculate_popularity(train_df):
    """Steam 게임별 소유 사용자 수를 기준으로 인기도 순위를 계산합니다."""

    popularity = (
        train_df.groupby("unified_game_id")["user_id"]
        .nunique()
        .reset_index(name="owner_count")
        .sort_values(
            ["owner_count", "unified_game_id"],
            ascending=[False, True]
        )
    )

    return popularity["unified_game_id"].tolist()


def recommend_games(popularity, observed_games, k=10):
    """이미 소유한 게임을 제외하고 인기 게임 상위 K개를 추천합니다."""

    recommendations = []

    for game_id in popularity:
        if game_id not in observed_games:
            recommendations.append(game_id)

            if len(recommendations) == k:
                break

    return recommendations


def evaluate_user(recommendations, heldout_games, k=10):
    """사용자 한 명의 Recall@K, NDCG@K, Hit@K를 계산합니다."""

    hits = [
        int(game_id in heldout_games)
        for game_id in recommendations[:k]
    ]

    # Recall@K
    recall = sum(hits) / len(heldout_games)

    # Hit@K
    hit = float(any(hits))

    # NDCG@K
    dcg = sum(
        relevance / math.log2(rank + 2)
        for rank, relevance in enumerate(hits)
    )

    idcg = sum(
        1 / math.log2(rank + 2)
        for rank in range(min(k, len(heldout_games)))
    )

    ndcg = dcg / idcg

    return recall, ndcg, hit


def evaluate_all_users(popularity, observed_df, heldout_df, k=10):
    """전체 평가 사용자의 지표를 계산하고 평균을 반환합니다."""

    observed_by_user = (
        observed_df.groupby("user_id")["unified_game_id"]
        .agg(set)
        .to_dict()
    )

    heldout_by_user = (
        heldout_df.groupby("user_id")["unified_game_id"]
        .agg(set)
        .to_dict()
    )

    if set(observed_by_user) != set(heldout_by_user):
        raise ValueError("Observed와 Heldout의 사용자 집합이 다릅니다.")

    total_recall = 0.0
    total_ndcg = 0.0
    total_hit = 0.0

    for user_id, observed_games in observed_by_user.items():
        heldout_games = heldout_by_user[user_id]

        if not heldout_games:
            raise ValueError(f"사용자 {user_id}의 Heldout이 비어 있습니다.")

        if not observed_games.isdisjoint(heldout_games):
            raise ValueError(f"사용자 {user_id}의 데이터가 중복됩니다.")

        recommendations = recommend_games(
            popularity, observed_games, k
        )

        recall, ndcg, hit = evaluate_user(
            recommendations, heldout_games, k
        )

        total_recall += recall
        total_ndcg += ndcg
        total_hit += hit

    n_users = len(observed_by_user)

    return {
        "n_users": n_users,
        "recall": total_recall / n_users,
        "ndcg": total_ndcg / n_users,
        "hit_rate": total_hit / n_users
    }


def main():
    # Steam 학습 데이터
    train_df = pd.read_parquet(
        DATA_DIR / "steam_train.parquet",
        columns=["user_id", "unified_game_id"]
    )

    # Steam 검증 데이터
    val_observed = pd.read_parquet(
        DATA_DIR / "steam_val_observed.parquet"
    )

    val_heldout = pd.read_parquet(
        DATA_DIR / "steam_val_heldout.parquet"
    )

    # 인기도 모델 구성
    popularity = calculate_popularity(train_df)

    # Steam Validation 평가
    results = evaluate_all_users(
        popularity,
        val_observed,
        val_heldout,
        k=TOP_K
    )

    # 결과 출력
    print("=" * 50)
    print("Popularity Baseline - Steam Validation")
    print("=" * 50)

    print(f"학습 사용자 수: {train_df['user_id'].nunique():,}명")
    print(f"추천 후보 게임: {len(popularity):,}개")
    print(f"평가 사용자 수: {results['n_users']:,}명")

    print(f"\nRecall@{TOP_K}: {results['recall'] * 100:.2f}%")
    print(f"NDCG@{TOP_K}: {results['ndcg'] * 100:.2f}%")
    print(f"Hit Rate@{TOP_K}: {results['hit_rate'] * 100:.2f}%")


if __name__ == "__main__":
    main()