"""Compare PlayStation observed-popularity and Steam-trained Weighted ALS at each rank.

This reproduces the existing PS test protocol without tuning parameters.
Run from repository root:
    python src/models/analyze_ps_weighted_rank_hits.py
"""

import numpy as np
import pandas as pd

import als_ps_test as evaluation
from analyze_ps_rank_hits import count_hits_at_ranks, top_unobserved_games


RESULT_PATH = evaluation.RESULT_DIR / "ps_weighted_rank_hit_comparison.csv"


def main():
    train = pd.read_parquet(
        evaluation.DATA_DIR / "steam_train.parquet",
        columns=["user_id", "unified_game_id", "playtime_forever"],
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

    observed_by_user = ps_observed.groupby("user_id")["unified_game_id"].agg(set).to_dict()
    heldout_by_user = ps_heldout.groupby("user_id")["unified_game_id"].agg(set).to_dict()
    if observed_by_user.keys() != heldout_by_user.keys():
        raise ValueError("PS observed and held-out user sets do not match")

    user_indices, game_indices, game_ids, game_to_idx, shape = (
        evaluation.prepare_train_data(train)
    )
    ps_matrix, ps_user_ids = evaluation.build_ps_matrix(
        ps_observed, game_ids, game_to_idx
    )

    positive_hours = train.loc[
        train["playtime_forever"] > 0, "playtime_forever"
    ].to_numpy(dtype=np.float64) / 60
    cap_hours = float(np.quantile(positive_hours, evaluation.PLAYTIME_PERCENTILE))
    print(f"Weighted ALS cap: {cap_hours:.6f} h; beta: {evaluation.PLAYTIME_BETA}")

    train_matrix = evaluation.build_train_matrix(
        train, user_indices, game_indices, shape,
        weighted=True, cap_hours=cap_hours,
    )
    weighted_model = evaluation.train_als(train_matrix, "Playtime Weighted ALS")

    popularity_ranking = evaluation.rank_games_by_popularity(ps_observed, game_ids)
    popularity_hits = np.zeros(evaluation.TOP_K, dtype=np.int64)
    weighted_hits = np.zeros(evaluation.TOP_K, dtype=np.int64)
    metric_sums = {
        "PS Popularity": {"recall_at_10": 0.0, "ndcg_at_10": 0.0, "hit_at_10": 0.0},
        "Playtime Weighted ALS": {"recall_at_10": 0.0, "ndcg_at_10": 0.0, "hit_at_10": 0.0},
    }

    num_users = len(ps_user_ids)
    for start in range(0, num_users, evaluation.BATCH_SIZE):
        end = min(start + evaluation.BATCH_SIZE, num_users)
        batch_matrix = ps_matrix[start:end]
        item_indices, _ = weighted_model.recommend(
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
                raise ValueError(f"{user_id}: observed and heldout overlap")

            popularity_recs = top_unobserved_games(
                popularity_ranking, observed_games, evaluation.TOP_K
            )
            weighted_recs = [game_ids[int(idx)] for idx in item_indices[local_index]]
            if not set(weighted_recs).isdisjoint(observed_games):
                raise ValueError(f"{user_id}: recommended already observed game")

            popularity_hits += count_hits_at_ranks(
                popularity_recs, heldout_games, evaluation.TOP_K
            )
            weighted_hits += count_hits_at_ranks(
                weighted_recs, heldout_games, evaluation.TOP_K
            )
            for model_name, recs in (
                ("PS Popularity", popularity_recs),
                ("Playtime Weighted ALS", weighted_recs),
            ):
                user_metrics = evaluation.evaluate_user(recs, heldout_games, evaluation.TOP_K)
                for name, value in user_metrics.items():
                    metric_sums[model_name][name] += value

        print(f"\rEvaluated: {end:,}/{num_users:,}", end="", flush=True)
    print()

    result = pd.DataFrame({
        "rank": np.arange(1, evaluation.TOP_K + 1),
        "ps_popularity_hits": popularity_hits,
        "ps_popularity_hit_rate": popularity_hits / num_users,
        "weighted_als_hits": weighted_hits,
        "weighted_als_hit_rate": weighted_hits / num_users,
    })
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULT_PATH, index=False)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"PS Popularity total hits: {popularity_hits.sum():,}")
    print(f"Weighted ALS total hits: {weighted_hits.sum():,}")
    for model_name, sums in metric_sums.items():
        print(model_name, {key: round(val / num_users, 8) for key, val in sums.items()})
    print(f"Saved {RESULT_PATH}")


if __name__ == "__main__":
    main()
