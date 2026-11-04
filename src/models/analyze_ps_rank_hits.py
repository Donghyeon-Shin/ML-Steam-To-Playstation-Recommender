"""Compare recommendation-position hit rates on the PlayStation test split.

This is a post-hoc diagnostic, not a new model-selection experiment. It uses
the same Steam-trained Binary ALS and PS-observed popularity baseline as
als_ps_test.py, without modifying the original evaluation results.

Run from the project root:
    python src/models/analyze_ps_rank_hits.py
"""

import als_ps_test as evaluation
import numpy as np
import pandas as pd


RESULT_PATH = evaluation.RESULT_DIR / "ps_rank_hit_comparison.csv"


def top_unobserved_games(ranked_games, observed_games, k):
    """Return the top-k popularity-ranked games absent from the user history."""
    recommendations = []

    for game_id in ranked_games:
        if game_id not in observed_games:
            recommendations.append(game_id)

            if len(recommendations) == k:
                break

    if len(recommendations) != k:
        raise ValueError("Too few unobserved games to fill the recommendation list.")

    return recommendations


def count_hits_at_ranks(recommended_games, heldout_games, k):
    """Return one binary hit indicator per recommendation position."""
    if len(recommended_games) != k or len(set(recommended_games)) != k:
        raise ValueError("Recommendations must contain exactly k unique games.")

    return np.fromiter(
        (game_id in heldout_games for game_id in recommended_games),
        count=k,
        dtype=np.int64,
    )


def main():
    train = pd.read_parquet(
        evaluation.DATA_DIR / "steam_train.parquet",
        columns=["user_id", "unified_game_id"],
    )
    ps_observed = pd.read_parquet(
        evaluation.DATA_DIR / "ps_test_observed.parquet",
        columns=["user_id", "unified_game_id"],
    )
    ps_heldout = pd.read_parquet(
        evaluation.DATA_DIR / "ps_test_heldout.parquet",
        columns=["user_id", "unified_game_id"],
    )

    for name, frame in (
        ("Steam Train", train),
        ("PS Observed", ps_observed),
        ("PS Heldout", ps_heldout),
    ):
        evaluation.validate_interactions(frame, name)

    observed_by_user = (
        ps_observed.groupby("user_id")["unified_game_id"].agg(set).to_dict()
    )
    heldout_by_user = (
        ps_heldout.groupby("user_id")["unified_game_id"].agg(set).to_dict()
    )

    if observed_by_user.keys() != heldout_by_user.keys():
        raise ValueError("PS observed and held-out user sets do not match.")

    user_indices, game_indices, game_ids, game_to_idx, shape = (
        evaluation.prepare_train_data(train)
    )
    ps_matrix, ps_user_ids = evaluation.build_ps_matrix(
        ps_observed, game_ids, game_to_idx
    )
    train_matrix = evaluation.build_train_matrix(
        train, user_indices, game_indices, shape, weighted=False
    )
    binary_model = evaluation.train_als(train_matrix, "Binary ALS")

    # PS popularity is calculated only from observed ownership, not held-out.
    popularity_ranking = evaluation.rank_games_by_popularity(ps_observed, game_ids)
    popularity_hits = np.zeros(evaluation.TOP_K, dtype=np.int64)
    binary_hits = np.zeros(evaluation.TOP_K, dtype=np.int64)

    num_users = len(ps_user_ids)

    for start in range(0, num_users, evaluation.BATCH_SIZE):
        end = min(start + evaluation.BATCH_SIZE, num_users)
        batch_matrix = ps_matrix[start:end]

        item_indices, _ = binary_model.recommend(
            userid=np.arange(end - start),
            user_items=batch_matrix,
            N=evaluation.TOP_K,
            filter_already_liked_items=True,
            recalculate_user=True,
        )

        for local_index, user_id in enumerate(ps_user_ids[start:end]):
            observed_games = observed_by_user[user_id]
            heldout_games = heldout_by_user[user_id]

            if not observed_games.isdisjoint(heldout_games):
                raise ValueError(f"{user_id}: observed and held-out games overlap.")

            popularity_recommendations = top_unobserved_games(
                popularity_ranking, observed_games, evaluation.TOP_K
            )
            binary_recommendations = [
                game_ids[int(index)] for index in item_indices[local_index]
            ]

            if not set(binary_recommendations).isdisjoint(observed_games):
                raise ValueError(f"{user_id}: Binary ALS recommended an observed game.")

            popularity_hits += count_hits_at_ranks(
                popularity_recommendations, heldout_games, evaluation.TOP_K
            )
            binary_hits += count_hits_at_ranks(
                binary_recommendations, heldout_games, evaluation.TOP_K
            )

        print(f"\rEvaluated users: {end:,}/{num_users:,}", end="", flush=True)

    print()

    results = pd.DataFrame(
        {
            "rank": np.arange(1, evaluation.TOP_K + 1),
            "ps_popularity_hits": popularity_hits,
            "ps_popularity_hit_rate": popularity_hits / num_users,
            "binary_als_hits": binary_hits,
            "binary_als_hit_rate": binary_hits / num_users,
        }
    )

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_PATH, index=False)

    print("\nPS recommendation-position hit rates")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"\nEvaluation users: {num_users:,}")
    print(f"PS Popularity total correct recommendations: {popularity_hits.sum():,}")
    print(f"Binary ALS total correct recommendations: {binary_hits.sum():,}")
    print(f"Results saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
