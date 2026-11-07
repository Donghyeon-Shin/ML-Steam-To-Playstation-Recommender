"""
Temporary Playtime Weighted ALS experiment using positive-playtime training only.

Run from the project root:
    python src/models/als_playtime_positive_only.py

Only Steam training interactions with playtime_forever == 0 are removed.
The original Weighted ALS script and Steam validation splits remain unchanged.
Results are written to a separate CSV for comparison with the original model.
"""

import os

# Limit BLAS threading consistently with the original training script.
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd

from als_playtime import (
    BATCH_SIZE,
    DATA_DIR,
    FACTORS,
    ITERATIONS,
    OWNER_ALPHA,
    PLAYTIME_BETA,
    PLAYTIME_PERCENTILE,
    PROJECT_DIR,
    RANDOM_STATE,
    REGULARIZATION,
    TOP_K,
    build_train_matrix,
    build_validation_matrix,
    calculate_playtime_cap,
    evaluate_all_users,
    train_als,
    validate_interactions,
    validate_train_playtime,
)


RESULT_PATH = PROJECT_DIR / "data" / "results" / "als_playtime_positive_only.csv"


def filter_positive_playtime(train_df):
    """Drop only zero-playtime Steam training interactions."""

    positive_train = train_df.loc[
        train_df["playtime_forever"] > 0
    ].copy()

    if positive_train.empty:
        raise ValueError("No positive-playtime training interactions remain.")

    return positive_train


def main():
    # Keep the data sources identical to the original Weighted ALS experiment.
    train_df = pd.read_parquet(
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

    validate_interactions(train_df, "Steam Train")
    validate_interactions(val_observed, "Steam Validation Observed")
    validate_interactions(val_heldout, "Steam Validation Heldout")
    train_df = validate_train_playtime(train_df)

    original_interactions = len(train_df)
    original_users = train_df["user_id"].nunique()
    original_games = train_df["unified_game_id"].nunique()

    # Change only the Steam training data; never alter validation labels.
    train_df = filter_positive_playtime(train_df)
    excluded_interactions = original_interactions - len(train_df)

    print("=" * 60)
    print("Playtime Weighted ALS - Positive-Playtime-Only Experiment")
    print("=" * 60)
    print(f"Original training interactions: {original_interactions:,}")
    print(f"Excluded zero-playtime interactions: {excluded_interactions:,}")
    print(f"Remaining training interactions: {len(train_df):,}")
    print(f"Training users: {original_users:,} -> {train_df['user_id'].nunique():,}")
    print(f"Training games: {original_games:,} -> {train_df['unified_game_id'].nunique():,}")
    print(f"Validation observed interactions: {len(val_observed):,}")
    print(f"Validation held-out interactions: {len(val_heldout):,}")

    # The cap is still determined only by positive Steam training playtimes.
    cap_hours = calculate_playtime_cap(train_df)
    train_matrix, game_ids, game_to_idx = build_train_matrix(
        train_df, cap_hours
    )

    print(f"Playtime beta: {PLAYTIME_BETA}")
    print(f"Playtime cap ({PLAYTIME_PERCENTILE:.0%}): {cap_hours:.2f} hours")
    print(f"Training matrix shape: {train_matrix.shape}")

    val_matrix, val_user_ids = build_validation_matrix(
        val_observed, game_ids, game_to_idx
    )
    no_known_observed = int((val_matrix.getnnz(axis=1) == 0).sum())
    known_heldout = int(
        val_heldout["unified_game_id"].isin(game_to_idx).sum()
    )

    print(f"Validation users without known observed games: {no_known_observed:,}")
    print(f"Held-out interactions in training catalog: {known_heldout:,}/{len(val_heldout):,}")

    # Use the same ALS hyperparameters, fold-in procedure, and metric functions.
    model = train_als(train_matrix)
    results_df = evaluate_all_users(
        model,
        val_matrix,
        val_user_ids,
        val_observed,
        val_heldout,
        game_ids,
        k=TOP_K,
        batch_size=BATCH_SIZE,
    )
    mean_metrics = results_df[
        ["recall_at_10", "ndcg_at_10", "hit_at_10"]
    ].mean()

    result = {
        "experiment": "positive_playtime_only",
        "factors": FACTORS,
        "regularization": REGULARIZATION,
        "iterations": ITERATIONS,
        "random_state": RANDOM_STATE,
        "owner_alpha": OWNER_ALPHA,
        "playtime_beta": PLAYTIME_BETA,
        "cap_percentile": PLAYTIME_PERCENTILE,
        "cap_hours": cap_hours,
        "original_train_interactions": original_interactions,
        "excluded_zero_interactions": excluded_interactions,
        "train_interactions": len(train_df),
        "train_users": train_matrix.shape[0],
        "candidate_games": len(game_ids),
        "validation_users": len(results_df),
        "validation_users_without_known_observed": no_known_observed,
        "heldout_interactions_in_catalog": known_heldout,
        **mean_metrics.to_dict(),
    }

    print("\n" + "=" * 60)
    print("Positive-Playtime-Only Validation Results")
    print("=" * 60)
    for metric in ["recall_at_10", "ndcg_at_10", "hit_at_10"]:
        print(f"{metric}: {result[metric]:.4f} ({result[metric] * 100:.2f}%)")

    # Never overwrite the original model's result files.
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(RESULT_PATH, index=False)
    print(f"Experiment summary saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
