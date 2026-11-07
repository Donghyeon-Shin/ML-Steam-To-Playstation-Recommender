
import os

# Limit OpenBLAS to one thread.
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from pathlib import Path
from itertools import product
from time import perf_counter

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from implicit.cpu.als import AlternatingLeastSquares

from als_binary import (
    build_train_matrix,
    group_games_by_user,
    evaluate_user
)


# ============================================================
# 1. Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULT_DIR = PROJECT_DIR / "data" / "results"

RESULT_PATH = RESULT_DIR / "als_binary_gridsearch.csv"

TOP_K = 10
BATCH_SIZE = 256
ITERATIONS = 40
RANDOM_STATE = 42

# Fine-grained factor comparison around the previously selected value.
FACTOR_GRID = [16, 18, 20, 22, 24]
REGULARIZATION_GRID = [5.0]
ALPHA_GRID = [2]


# ============================================================
# 2. Validation evaluation
# ============================================================

def evaluate_model(
    model,
    observed_by_user,
    heldout_by_user,
    game_ids,
    game_to_idx,
    confidence_alpha
):
    """
    Evaluate all Steam validation users in batches.

    Fold in new users using their observed games and evaluate
    top-10 recommendations.
    """

    user_ids = list(observed_by_user.keys())
    num_users = len(user_ids)
    num_games = len(game_ids)

    metric_sums = {
        "recall_at_10": 0.0,
        "ndcg_at_10": 0.0,
        "hit_at_10": 0.0
    }

    # Process users in batches.
    for start in range(0, num_users, BATCH_SIZE):

        batch_users = user_ids[start:start + BATCH_SIZE]

        rows = []
        cols = []

        # Build the batch user-by-game matrix.
        for local_idx, user_id in enumerate(batch_users):

            observed_games = observed_by_user[user_id]

            for game_id in observed_games:

                game_idx = game_to_idx.get(game_id)

                if game_idx is not None:
                    rows.append(local_idx)
                    cols.append(game_idx)

        # Binary Confidence
        confidence = np.full(
            len(rows),
            1 + confidence_alpha,
            dtype=np.float32
        )

        # CSR stores only observed games instead of a dense user-game array.
        # confidence holds weights; rows are batch-local user positions and
        # cols are the corresponding training-game column positions.
        # shape defines all batch users and training-game candidates.
        batch_matrix = csr_matrix(
            (
                confidence,
                (rows, cols)
            ),
            shape=(len(batch_users), num_games),
            dtype=np.float32
        )

        # Fold in validation users against the game factors learned on Steam.
        # recalculate_user=True derives factors from each user's observed games.
        item_indices, scores = model.recommend(
            userid=np.arange(len(batch_users)),  # Batch-local CSR row indices
            user_items=batch_matrix,              # Observed-game confidence
            N=TOP_K,                              # Games to return per user
            filter_already_liked_items=True,     # Exclude observed games
            recalculate_user=True                # Compute new user factors
        )
        # Returns matching arrays of ranked game-column indices and scores.

        # Evaluate each user.
        for row, user_id in enumerate(batch_users):

            observed_games = observed_by_user[user_id]
            heldout_games = heldout_by_user[user_id]

            recommendations = [
                (
                    game_ids[int(item_idx)],
                    float(score)
                )
                for item_idx, score in zip(
                    item_indices[row],
                    scores[row]
                )
            ]

            recommended_games = {
                game_id
                for game_id, _ in recommendations
            }

            # Check for observed/held-out overlap.
            if not observed_games.isdisjoint(heldout_games):
                raise ValueError(
                    f"User {user_id}: overlapping observed and held-out games"
                )

            # Ensure observed games were not recommended.
            if not recommended_games.isdisjoint(observed_games):
                raise ValueError(
                    f"User {user_id}: an observed game was recommended"
                )

            # Check for duplicate recommendations.
            if len(recommended_games) != len(recommendations):
                raise ValueError(
                    f"User {user_id}: duplicate recommendations"
                )

            # Evaluate recommendations.
            metrics = evaluate_user(
                recommendations,
                heldout_games,
                k=TOP_K
            )

            for metric_name in metric_sums:
                metric_sums[metric_name] += metrics[metric_name]

    # Return the mean of per-user metrics.
    return {
        metric_name: total / num_users
        for metric_name, total in metric_sums.items()
    }


# ============================================================
# 3. Load existing experiment results
# ============================================================

def load_existing_results():
    """
    Load saved CSV results and return completed hyperparameter combinations.
    """

    if not RESULT_PATH.exists():
        return pd.DataFrame(), set()

    if RESULT_PATH.stat().st_size == 0:
        return pd.DataFrame(), set()

    existing_df = pd.read_csv(
        RESULT_PATH,
        encoding="utf-8-sig"
    )

    completed_params = {
        (
            int(row.factors),
            float(row.regularization),
            int(row.confidence_alpha),
            int(row.iterations)
        )
        for row in existing_df.itertuples(index=False)
    }

    return existing_df, completed_params


# ============================================================
# 4. Grid Search
# ============================================================

def grid_search(
    train_matrix,
    observed_by_user,
    heldout_by_user,
    game_ids,
    game_to_idx
):
    """
    Run the grid search.

    Skip combinations already saved in the CSV, and append only new results.
    """

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Check existing results.
    existing_df, completed_params = load_existing_results()

    # Generate all hyperparameter combinations.
    combinations = list(product(
        FACTOR_GRID,
        REGULARIZATION_GRID,
        ALPHA_GRID
    ))

    total_combinations = len(combinations)

    # Filter out previously completed combinations.
    remaining_combinations = [
        params
        for params in combinations
        if (
            params[0],
            params[1],
            params[2],
            ITERATIONS
        ) not in completed_params
    ]

    print("\n" + "=" * 60)
    print("Binary ALS Grid Search")
    print("=" * 60)

    print(f"Existing experiments: {len(existing_df)}")
    print(f"Search combinations: {total_combinations}")
    print(f"Remaining combinations: {len(remaining_combinations)}")
    print(f"Evaluation users: {len(observed_by_user):,}")
    print(f"Recommendations: Top-{TOP_K}")
    print(f"Iterations: {ITERATIONS}")

    total_start_time = perf_counter()

    # Run only new experiments.
    for experiment_idx, (
        factors,
        regularization,
        confidence_alpha
    ) in enumerate(remaining_combinations, start=1):

        print("\n" + "-" * 60)
        print(
            f"New experiment {experiment_idx}/"
            f"{len(remaining_combinations)}"
        )
        print("-" * 60)

        print(f"Factors: {factors}")
        print(f"Regularization: {regularization}")
        print(f"Confidence Alpha: {confidence_alpha}")

        start_time = perf_counter()

        # ----------------------------------------------------
        # 1. Update confidence values.
        # ----------------------------------------------------

        train_matrix.data.fill(
            1 + confidence_alpha
        )

        # ----------------------------------------------------
        # 2. Create and train the model.
        # ----------------------------------------------------

        # The implicit CPU ALS class learns user/game latent factor matrices.
        # API: https://benfred.github.io/implicit/api/models/cpu/als.html
        model = AlternatingLeastSquares(
            factors=factors,                # Number of latent dimensions
            regularization=regularization,  # L2 penalty against overfitting
            iterations=ITERATIONS,          # ALS optimization passes
            alpha=1.0,                     # Preserve CSR confidence values
            random_state=RANDOM_STATE,     # Seed factor initialization
            dtype=np.float32               # Use 32-bit factor vectors
        )

        # fit() factorizes the user-by-game confidence matrix; absent entries
        # represent unobserved games. Hide progress for grid search trials.
        model.fit(
            train_matrix,        # CSR with confidence for owned games
            show_progress=False  # Suppress per-iteration progress bar
        )

        # ----------------------------------------------------
        # 3. Evaluate on the full validation split.
        # ----------------------------------------------------

        metrics = evaluate_model(
            model,
            observed_by_user,
            heldout_by_user,
            game_ids,
            game_to_idx,
            confidence_alpha
        )

        elapsed_seconds = perf_counter() - start_time

        # ----------------------------------------------------
        # 4. Record results.
        # ----------------------------------------------------

        result = {
            "factors": factors,
            "regularization": regularization,
            "confidence_alpha": confidence_alpha,
            "iterations": ITERATIONS,
            "recall_at_10": metrics["recall_at_10"],
            "ndcg_at_10": metrics["ndcg_at_10"],
            "hit_at_10": metrics["hit_at_10"],
            "elapsed_seconds": elapsed_seconds
        }

        print(
            f"\nRecall@10: "
            f"{metrics['recall_at_10'] * 100:.2f}%"
        )

        print(
            f"NDCG@10: "
            f"{metrics['ndcg_at_10'] * 100:.2f}%"
        )

        print(
            f"Hit Rate@10: "
            f"{metrics['hit_at_10'] * 100:.2f}%"
        )

        print(f"Elapsed time: {elapsed_seconds:.2f} seconds")

        # ----------------------------------------------------
        # 5. Append only new results to the CSV.
        # ----------------------------------------------------

        file_exists = (
            RESULT_PATH.exists()
            and RESULT_PATH.stat().st_size > 0
        )

        pd.DataFrame([result]).to_csv(
            RESULT_PATH,
            mode="a",
            index=False,
            header=not file_exists,
            encoding="utf-8"
        )

        # Mark the combination as completed.
        completed_params.add((
            factors,
            regularization,
            confidence_alpha,
            ITERATIONS
        ))

        print("CSV saved")

    total_elapsed = perf_counter() - total_start_time

    print("\n" + "=" * 60)
    print("Grid Search completed")
    print("=" * 60)

    print(f"Current run time: {total_elapsed:.2f} seconds")

    # Reload both existing and newly added results.
    all_results = pd.read_csv(
        RESULT_PATH,
        encoding="utf-8-sig"
    )

    print(f"Total accumulated experiments: {len(all_results)}")

    return all_results


# ============================================================
# 5. Display the best hyperparameters
# ============================================================

def print_best_result(results_df):
    """
    Sort all accumulated results by NDCG@10 and display
    the best hyperparameters.
    """

    sorted_results = (
        results_df.sort_values(
            by=[
                "ndcg_at_10",
                "recall_at_10",
                "hit_at_10"
            ],
            ascending=[False, False, False]
        )
        .reset_index(drop=True)
    )

    best = sorted_results.iloc[0]

    print("\n" + "=" * 60)
    print("Final Grid Search Results")
    print("=" * 60)

    print("\nTop 10 combinations by NDCG@10")

    top_results = sorted_results[
        [
            "factors",
            "regularization",
            "confidence_alpha",
            "recall_at_10",
            "ndcg_at_10",
            "hit_at_10"
        ]
    ].head(10).copy()

    # Convert metrics to percentages.
    for column in [
        "recall_at_10",
        "ndcg_at_10",
        "hit_at_10"
    ]:
        top_results[column] *= 100

    print(
        top_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n" + "=" * 60)
    print("Best Hyperparameters")
    print("=" * 60)

    print(f"Factors: {int(best['factors'])}")

    print(
        f"Regularization: "
        f"{best['regularization']}"
    )

    print(
        f"Confidence Alpha: "
        f"{int(best['confidence_alpha'])}"
    )

    print(
        f"Iterations: "
        f"{int(best['iterations'])}"
    )

    print("\nBest Model Evaluation Metrics")

    print(
        f"Recall@10: "
        f"{best['recall_at_10'] * 100:.2f}%"
    )

    print(
        f"NDCG@10: "
        f"{best['ndcg_at_10'] * 100:.2f}%"
    )

    print(
        f"Hit Rate@10: "
        f"{best['hit_at_10'] * 100:.2f}%"
    )

    print(f"\nFull results saved to: {RESULT_PATH}")


# ============================================================
# 6. Main execution
# ============================================================

def main():

    # 1. Load data.
    train_df = pd.read_parquet(
        DATA_DIR / "steam_train.parquet",
        columns=["user_id", "unified_game_id"]
    )

    val_observed = pd.read_parquet(
        DATA_DIR / "steam_val_observed.parquet",
        columns=["user_id", "unified_game_id"]
    )

    val_heldout = pd.read_parquet(
        DATA_DIR / "steam_val_heldout.parquet",
        columns=["user_id", "unified_game_id"]
    )

    print("Steam data loaded")
    print(f"Training interactions: {len(train_df):,}")

    # 2. Build the training matrix.
    train_matrix, game_ids, game_to_idx = (
        build_train_matrix(train_df)
    )

    print(f"Training matrix: {train_matrix.shape}")

    # 3. Group validation games by user.
    observed_by_user = group_games_by_user(val_observed)
    heldout_by_user = group_games_by_user(val_heldout)

    # Verify that both validation splits contain the same users.
    if set(observed_by_user) != set(heldout_by_user):
        raise ValueError(
            "Observed and held-out user sets do not match."
        )

    # 4. Run the grid search.
    results_df = grid_search(
        train_matrix,
        observed_by_user,
        heldout_by_user,
        game_ids,
        game_to_idx
    )

    # 5. Select the best model from accumulated results.
    print_best_result(results_df)


if __name__ == "__main__":
    main()
