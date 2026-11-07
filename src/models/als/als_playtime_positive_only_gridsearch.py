"""
Small beta grid search for Playtime Weighted ALS without zero-playtime training.

Run from the project root:
    python src/models/als/als_playtime_positive_only_gridsearch.py

The beta=0 run is the binary-confidence baseline on the *filtered* dataset.
Steam validation observed/held-out splits remain unchanged for every beta.
This experiment does not modify the original Weighted ALS grid-search results.
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from time import perf_counter

import numpy as np
import pandas as pd

from implicit.cpu.als import AlternatingLeastSquares

from als_playtime import (
    BATCH_SIZE,
    DATA_DIR,
    FACTORS,
    ITERATIONS,
    OWNER_ALPHA,
    PLAYTIME_PERCENTILE,
    PROJECT_DIR,
    RANDOM_STATE,
    REGULARIZATION,
    TOP_K,
    build_validation_matrix,
    calculate_playtime_cap,
    evaluate_all_users,
    validate_interactions,
    validate_train_playtime,
)
from als_playtime_gridsearch import (
    build_train_matrix,
    prepare_train_data,
)
from als_playtime_positive_only import filter_positive_playtime


# Coarse sweep followed by 0.5 and 0.25-step refinement near the peak.
BETA_GRID = [
    0.0, 0.1, 0.25, 0.5, 1.0, 2.0,
    2.5, 3.0, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75,
    5.0, 5.25, 5.5, 6.0, 7.0,
    8.0, 16.0, 32.0, 64.0,
]
RESULT_PATH = (
    PROJECT_DIR / "data" / "results"
    / "als_playtime_positive_only_gridsearch.csv"
)


def main():
    # Keep the same training/validation split and ALS configuration.
    original_train = pd.read_parquet(
        DATA_DIR / "steam_train.parquet",
        columns=["user_id", "unified_game_id", "playtime_forever"],
    )
    val_observed = pd.read_parquet(
        DATA_DIR / "steam_val_observed.parquet",
        columns=["user_id", "unified_game_id"],
    )
    val_heldout = pd.read_parquet(
        DATA_DIR / "steam_val_heldout.parquet",
        columns=["user_id", "unified_game_id"],
    )

    validate_interactions(original_train, "Steam Train")
    validate_interactions(val_observed, "Steam Validation Observed")
    validate_interactions(val_heldout, "Steam Validation Heldout")
    original_train = validate_train_playtime(original_train)

    train_df = filter_positive_playtime(original_train)
    excluded_zero = len(original_train) - len(train_df)
    cap_hours = calculate_playtime_cap(train_df)

    (
        user_indices,
        game_indices,
        playtime_hours,
        matrix_shape,
        game_ids,
        game_to_idx,
    ) = prepare_train_data(train_df)

    # Compute the capped log-playtime term once for every beta.
    log_playtime = np.log1p(np.minimum(playtime_hours, cap_hours))
    val_matrix, val_user_ids = build_validation_matrix(
        val_observed, game_ids, game_to_idx
    )

    print("\n" + "=" * 60)
    print("Positive-Playtime-Only ALS - Beta Grid Search")
    print("=" * 60)
    print(f"Excluded zero-playtime training interactions: {excluded_zero:,}")
    print(f"Remaining training interactions: {len(train_df):,}")
    print(f"Training matrix shape: {matrix_shape}")
    print(f"Validation users: {len(val_user_ids):,}")
    print(f"Cap percentile: {PLAYTIME_PERCENTILE:.0%}")
    print(f"Cap hours: {cap_hours:.2f}")
    print(f"Beta values: {BETA_GRID}")

    # Reuse results from identical configurations on subsequent runs.
    rows = []
    if RESULT_PATH.exists() and RESULT_PATH.stat().st_size > 0:
        existing = pd.read_csv(RESULT_PATH)
        expected = {
            "factors": FACTORS,
            "regularization": REGULARIZATION,
            "iterations": ITERATIONS,
            "random_state": RANDOM_STATE,
            "owner_alpha": OWNER_ALPHA,
            "cap_percentile": PLAYTIME_PERCENTILE,
            "cap_hours": cap_hours,
            "excluded_zero_interactions": excluded_zero,
            "train_interactions": len(train_df),
            "train_users": matrix_shape[0],
            "candidate_games": matrix_shape[1],
            "validation_users": len(val_user_ids),
        }
        for column, value in expected.items():
            if column not in existing or not np.allclose(
                existing[column], value
            ):
                raise ValueError(
                    f"Existing grid-search CSV has different {column} values."
                )
        rows = existing.to_dict("records")

    for i, beta in enumerate(BETA_GRID, start=1):
        if any(np.isclose(row["playtime_beta"], beta) for row in rows):
            print(f"Skipping completed experiment: beta={beta}")
            continue

        print("\n" + "-" * 60)
        print(f"Experiment {i}/{len(BETA_GRID)}: beta={beta}")
        print("-" * 60)
        started = perf_counter()

        train_matrix = build_train_matrix(
            user_indices,
            game_indices,
            log_playtime,
            matrix_shape,
            beta,
        )
        model = AlternatingLeastSquares(
            factors=FACTORS,
            regularization=REGULARIZATION,
            iterations=ITERATIONS,
            alpha=1.0,
            random_state=RANDOM_STATE,
            dtype=np.float32,
        )
        model.fit(train_matrix, show_progress=False)

        user_results = evaluate_all_users(
            model,
            val_matrix,
            val_user_ids,
            val_observed,
            val_heldout,
            game_ids,
            k=TOP_K,
            batch_size=BATCH_SIZE,
        )
        metrics = user_results[
            ["recall_at_10", "ndcg_at_10", "hit_at_10"]
        ].mean()
        row = {
            "experiment": "positive_playtime_only",
            "factors": FACTORS,
            "regularization": REGULARIZATION,
            "iterations": ITERATIONS,
            "random_state": RANDOM_STATE,
            "owner_alpha": OWNER_ALPHA,
            "playtime_beta": beta,
            "cap_percentile": PLAYTIME_PERCENTILE,
            "cap_hours": cap_hours,
            "excluded_zero_interactions": excluded_zero,
            "train_interactions": len(train_df),
            "train_users": matrix_shape[0],
            "candidate_games": matrix_shape[1],
            "validation_users": len(user_results),
            **metrics.to_dict(),
            "elapsed_seconds": perf_counter() - started,
        }
        rows.append(row)

        print(
            f"beta={beta}: "
            f"Recall@10={row['recall_at_10']:.4f}, "
            f"NDCG@10={row['ndcg_at_10']:.4f}, "
            f"Hit@10={row['hit_at_10']:.4f}"
        )

        # Checkpoint completed experiments without touching original results.
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        (
            pd.DataFrame(rows)
            .sort_values("playtime_beta")
            .to_csv(RESULT_PATH, index=False)
        )
        print(f"Results saved to: {RESULT_PATH}")

    print("\n" + "=" * 60)
    print("Grid Search Summary (ranked by NDCG@10)")
    print("=" * 60)
    summary = (
        pd.DataFrame(rows)
        .sort_values("ndcg_at_10", ascending=False)
        [["playtime_beta", "recall_at_10", "ndcg_at_10", "hit_at_10"]]
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
