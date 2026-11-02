from pathlib import Path
import math

import pandas as pd


DATA_DIR = Path("data/processed")
TOP_K = 10


def calculate_popularity(train_df):
    """Rank Steam games by distinct owner count, breaking ties by game ID."""

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
    """Recommend the top-k popular games not owned by the user."""

    recommendations = []

    for game_id in popularity:
        if game_id not in observed_games:
            recommendations.append(game_id)

            if len(recommendations) == k:
                break

    return recommendations


def evaluate_user(recommendations, heldout_games, k=10):
    """Compute Recall@K, NDCG@K, and Hit@K for one user."""

    hits = [
        int(game_id in heldout_games)
        for game_id in recommendations[:k]
    ]

    # Recall@K: fraction of held-out games retrieved.
    recall = sum(hits) / len(heldout_games)

    # Hit@K: whether any held-out game was retrieved.
    hit = float(any(hits))

    # NDCG@K: discounted gains normalized by the ideal ranking.
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
    """Evaluate all users and return macro-averaged ranking metrics."""

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
        raise ValueError("Observed and held-out user sets do not match.")

    total_recall = 0.0
    total_ndcg = 0.0
    total_hit = 0.0

    for user_id, observed_games in observed_by_user.items():
        heldout_games = heldout_by_user[user_id]

        if not heldout_games:
            raise ValueError(f"User {user_id} has no held-out games.")

        if not observed_games.isdisjoint(heldout_games):
            raise ValueError(
                f"User {user_id} has overlapping observed and held-out games."
            )

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
    # Load Steam training interactions.
    train_df = pd.read_parquet(
        DATA_DIR / "steam_train.parquet",
        columns=["user_id", "unified_game_id"]
    )

    # Load Steam validation interactions.
    val_observed = pd.read_parquet(
        DATA_DIR / "steam_val_observed.parquet"
    )

    val_heldout = pd.read_parquet(
        DATA_DIR / "steam_val_heldout.parquet"
    )

    # Build the popularity ranking.
    popularity = calculate_popularity(train_df)

    # Evaluate on the Steam validation split.
    results = evaluate_all_users(
        popularity,
        val_observed,
        val_heldout,
        k=TOP_K
    )

    # Report evaluation results.
    print("=" * 50)
    print("Popularity Baseline - Steam Validation")
    print("=" * 50)

    print(f"Training users: {train_df['user_id'].nunique():,}")
    print(f"Candidate games: {len(popularity):,}")
    print(f"Evaluation users: {results['n_users']:,}")

    print(f"\nRecall@{TOP_K}: {results['recall'] * 100:.2f}%")
    print(f"NDCG@{TOP_K}: {results['ndcg'] * 100:.2f}%")
    print(f"Hit Rate@{TOP_K}: {results['hit_rate'] * 100:.2f}%")


if __name__ == "__main__":
    main()
