
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

from als_playtime import (
    OWNER_ALPHA,
    build_validation_matrix,
    evaluate_all_users,
    validate_interactions,
    validate_train_playtime
)


# ============================================================
# 1. Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULT_DIR = PROJECT_DIR / "data" / "results"

RESULT_PATH = RESULT_DIR / "als_playtime_gridsearch.csv"

TOP_K = 10
BATCH_SIZE = 256
RANDOM_STATE = 42

# ALS hyperparameters
FACTOR_GRID = [24]
REGULARIZATION_GRID = [5.0]
ITERATION_GRID = [40]

# Playtime weight values
PLAYTIME_BETA_GRID = [
    0.1,
    0.25,
    0.5,
    1.0
]

# Playtime cap percentiles
PLAYTIME_CAP_PERCENTILE_GRID = [
    0.50,
    0.75,
    0.90,
    0.95,
    0.99
]


# ============================================================
# 2. Training data preparation
# ============================================================

def prepare_train_data(train_df):
    """
    Create user and game indices for Steam Train.

    Convert raw playtimes to hours for later use.
    """

    # factorize() returns integer codes and distinct original IDs. Codes
    # become row indices; user_ids maps rows back to Steam user IDs.
    # sort=True sorts unique IDs while preserving their code mapping.
    user_indices, user_ids = pd.factorize(
        train_df["user_id"],
        sort=True
    )

    # Game codes become column indices; game_ids is the reverse lookup.
    game_indices, game_ids = pd.factorize(
        train_df["unified_game_id"],
        sort=True
    )

    # Convert minutes to hours.
    playtime_hours = (
        train_df["playtime_forever"]
        .to_numpy(dtype=np.float64) / 60
    )

    # Map game IDs to matrix column indices.
    game_to_idx = {
        game_id: idx
        for idx, game_id in enumerate(game_ids)
    }

    matrix_shape = (
        len(user_ids),
        len(game_ids)
    )

    return (
        user_indices,
        game_indices,
        playtime_hours,
        matrix_shape,
        game_ids,
        game_to_idx
    )


# ============================================================
# 3. Calculate candidate playtime caps
# ============================================================

def calculate_cap_grid(playtime_hours):
    """
    Calculate percentile-based caps from positive Steam Train playtimes.
    """

    positive_hours = playtime_hours[
        playtime_hours > 0
    ]

    if len(positive_hours) == 0:
        raise ValueError(
            "The training data contain no positive playtime values."
        )

    cap_grid = []

    for percentile in PLAYTIME_CAP_PERCENTILE_GRID:

        cap_hours = float(
            np.quantile(
                positive_hours,
                percentile
            )
        )

        cap_grid.append((
            percentile,
            cap_hours
        ))

    return cap_grid


# ============================================================
# 4. Weighted ALS training matrix construction
# ============================================================

def build_train_matrix(
    user_indices,
    game_indices,
    log_playtime,
    matrix_shape,
    beta
):
    """
    Build a training CSR matrix from beta and log-transformed playtimes.

    c_ui = 1 + OWNER_ALPHA + beta * log_playtime
    """

    confidence = (
        1
        + OWNER_ALPHA
        + beta * log_playtime
    ).astype(np.float32)

    # CSR avoids storing the many unobserved user-game pairs as dense zeros.
    # log-playtime-weighted confidence[i] is stored at the row/column pair
    # (user_indices[i], game_indices[i]); matrix_shape fixes both dimensions
    # and dtype specifies 32-bit floating-point confidence values.
    train_matrix = csr_matrix(
        (
            confidence,
            (
                user_indices,
                game_indices
            )
        ),
        shape=matrix_shape,
        dtype=np.float32
    )

    return train_matrix


# ============================================================
# 5. Load existing experiment results
# ============================================================

def make_param_key(
    factors,
    regularization,
    iterations,
    owner_alpha,
    beta,
    cap_hours
):
    """
    Build a parameter key for identifying completed experiments.

    Round floating-point values to avoid comparison discrepancies.
    """

    return (
        int(factors),
        round(float(regularization), 8),
        int(iterations),
        int(owner_alpha),
        round(float(beta), 8),
        round(float(cap_hours), 6)
    )


def load_existing_results():
    """
    Load the existing CSV and identify completed parameter combinations.
    """

    if not RESULT_PATH.exists():
        return pd.DataFrame(), set()

    if RESULT_PATH.stat().st_size == 0:
        return pd.DataFrame(), set()

    existing_df = pd.read_csv(
        RESULT_PATH,
        encoding="utf-8-sig"
    )

    required_columns = {
        "factors",
        "regularization",
        "iterations",
        "owner_alpha",
        "playtime_beta",
        "cap_hours"
    }

    missing = required_columns - set(existing_df.columns)

    if missing:
        raise ValueError(
            f"Existing CSV is missing required columns: {missing}"
        )

    completed_params = {
        make_param_key(
            row.factors,
            row.regularization,
            row.iterations,
            row.owner_alpha,
            row.playtime_beta,
            row.cap_hours
        )
        for row in existing_df.itertuples(index=False)
    }

    return existing_df, completed_params


# ============================================================
# 6. Grid Search
# ============================================================

def grid_search(
    train_data,
    val_matrix,
    val_user_ids,
    val_observed,
    val_heldout,
    cap_grid
):
    """
    Search over beta values and playtime caps together.

    Skip completed experiments and append only new results to the CSV.
    """

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        user_indices,
        game_indices,
        playtime_hours,
        matrix_shape,
        game_ids,
        _
    ) = train_data

    # Check existing experiment results.
    existing_df, completed_params = load_existing_results()

    # --------------------------------------------------------
    # 1. Generate all experiment combinations.
    # --------------------------------------------------------

    combinations = []

    for percentile, cap_hours in cap_grid:

        for (
            factors,
            regularization,
            iterations,
            beta
        ) in product(
            FACTOR_GRID,
            REGULARIZATION_GRID,
            ITERATION_GRID,
            PLAYTIME_BETA_GRID
        ):

            combinations.append((
                percentile,
                cap_hours,
                factors,
                regularization,
                iterations,
                beta
            ))

    # --------------------------------------------------------
    # 2. Exclude completed combinations.
    # --------------------------------------------------------

    remaining = []

    for combination in combinations:

        (
            percentile,
            cap_hours,
            factors,
            regularization,
            iterations,
            beta
        ) = combination

        param_key = make_param_key(
            factors,
            regularization,
            iterations,
            OWNER_ALPHA,
            beta,
            cap_hours
        )

        if param_key not in completed_params:
            remaining.append(combination)

    print("\n" + "=" * 60)
    print("Playtime Weighted ALS Grid Search")
    print("=" * 60)

    print(f"Existing experiments: {len(existing_df)}")
    print(f"Search combinations: {len(combinations)}")
    print(f"Remaining combinations: {len(remaining)}")
    print(f"Evaluation users: {len(val_user_ids):,}")

    total_start = perf_counter()

    # Cache the most recently computed cap transformation.
    current_cap = None
    log_playtime = None

    # --------------------------------------------------------
    # 3. Run experiments.
    # --------------------------------------------------------

    for experiment_idx, combination in enumerate(
        remaining,
        start=1
    ):

        (
            percentile,
            cap_hours,
            factors,
            regularization,
            iterations,
            beta
        ) = combination

        print("\n" + "-" * 60)
        print(
            f"Experiment {experiment_idx}/{len(remaining)}"
        )
        print("-" * 60)

        print(f"Factors: {factors}")
        print(f"Regularization: {regularization}")
        print(f"Iterations: {iterations}")
        print(f"Owner Alpha: {OWNER_ALPHA}")
        print(f"Playtime Beta: {beta}")

        print(
            f"Cap Percentile: "
            f"{percentile * 100:.0f}%"
        )

        print(f"Cap Hours: {cap_hours:.2f}")

        start_time = perf_counter()

        # ----------------------------------------------------
        # 4. Compute log-playtime values for the current cap.
        # ----------------------------------------------------

        # Recompute only when the cap changes.
        if current_cap != cap_hours:

            clipped_hours = np.minimum(
                playtime_hours,
                cap_hours
            )

            log_playtime = np.log1p(
                clipped_hours
            )

            current_cap = cap_hours

        # ----------------------------------------------------
        # 5. Build the training matrix for each beta.
        # ----------------------------------------------------

        train_matrix = build_train_matrix(
            user_indices,
            game_indices,
            log_playtime,
            matrix_shape,
            beta
        )

        # ----------------------------------------------------
        # 6. Create and train the ALS model.
        # ----------------------------------------------------

        # The implicit CPU ALS class factorizes confidence-weighted user/game
        # interactions; one model is trained for each grid-search combination.
        # API: https://benfred.github.io/implicit/api/models/cpu/als.html
        model = AlternatingLeastSquares(
            factors=factors,                # Number of latent dimensions
            regularization=regularization,  # L2 penalty against overfitting
            iterations=iterations,          # ALS optimization passes
            alpha=1.0,                     # Keep preweighted CSR confidence
            random_state=RANDOM_STATE,     # Seed factor initialization
            dtype=np.float32               # Use 32-bit factor vectors
        )

        # fit() learns user/game latent factors from the CSR confidence matrix.
        # Missing interactions have zero preference and baseline confidence 1.
        model.fit(
            train_matrix,        # User-by-game weighted confidence CSR
            show_progress=False  # Suppress progress during grid search
        )

        # ----------------------------------------------------
        # 7. Evaluate on Steam validation.
        # ----------------------------------------------------

        results_df = evaluate_all_users(
            model,
            val_matrix,
            val_user_ids,
            val_observed,
            val_heldout,
            game_ids,
            k=TOP_K,
            batch_size=BATCH_SIZE
        )

        mean_metrics = results_df[
            [
                "recall_at_10",
                "ndcg_at_10",
                "hit_at_10"
            ]
        ].mean()

        elapsed_seconds = perf_counter() - start_time

        # ----------------------------------------------------
        # 8. Record results.
        # ----------------------------------------------------

        # Preserve the existing CSV column structure.
        result = {
            "factors": factors,
            "regularization": regularization,
            "iterations": iterations,
            "owner_alpha": OWNER_ALPHA,
            "playtime_beta": beta,
            "cap_hours": cap_hours,
            "recall_at_10": mean_metrics["recall_at_10"],
            "ndcg_at_10": mean_metrics["ndcg_at_10"],
            "hit_at_10": mean_metrics["hit_at_10"],
            "elapsed_seconds": elapsed_seconds
        }

        print("\nExperiment Results")

        print(
            f"Recall@10: "
            f"{mean_metrics['recall_at_10'] * 100:.2f}%"
        )

        print(
            f"NDCG@10: "
            f"{mean_metrics['ndcg_at_10'] * 100:.2f}%"
        )

        print(
            f"Hit Rate@10: "
            f"{mean_metrics['hit_at_10'] * 100:.2f}%"
        )

        print(f"Elapsed time: {elapsed_seconds:.2f} seconds")

        # ----------------------------------------------------
        # 9. Append results to the existing CSV.
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

        # Mark the parameter combination as completed.
        completed_params.add(
            make_param_key(
                factors,
                regularization,
                iterations,
                OWNER_ALPHA,
                beta,
                cap_hours
            )
        )

        print("CSV saved")

    # --------------------------------------------------------
    # 10. Reload accumulated results.
    # --------------------------------------------------------

    total_elapsed = perf_counter() - total_start

    all_results = pd.read_csv(
        RESULT_PATH,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("Grid Search completed")
    print("=" * 60)

    print(f"Current run time: {total_elapsed:.2f} seconds")
    print(f"Total accumulated experiments: {len(all_results)}")

    return all_results


# ============================================================
# 7. Display the best hyperparameters
# ============================================================

def print_best_result(results_df):
    """
    Display the best results ranked by NDCG@10.

    Since beta=0 is equivalent to Binary ALS, also report
    the best Weighted ALS result with beta>0.
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

    print("\n" + "=" * 60)
    print("Final Grid Search Results")
    print("=" * 60)

    print("\nTop 10 combinations by NDCG@10")

    display_df = sorted_results[
        [
            "factors",
            "regularization",
            "iterations",
            "playtime_beta",
            "cap_hours",
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
        display_df[column] *= 100

    print(
        display_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # Best Weighted ALS result with beta>0.
    # --------------------------------------------------------

    weighted_results = sorted_results[
        sorted_results["playtime_beta"] > 0
    ]

    if weighted_results.empty:
        print("\nNo experiments with beta>0 were found.")
        return

    best = weighted_results.iloc[0]

    print("\n" + "=" * 60)
    print("Best Playtime Weighted ALS Hyperparameters")
    print("=" * 60)

    print(f"Factors: {int(best['factors'])}")
    print(f"Regularization: {best['regularization']}")
    print(f"Iterations: {int(best['iterations'])}")
    print(f"Owner Alpha: {int(best['owner_alpha'])}")
    print(f"Playtime Beta: {best['playtime_beta']}")
    print(f"Playtime Cap: {best['cap_hours']:.2f} hours")

    print("\nBest Weighted ALS Evaluation Metrics")

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

    # --------------------------------------------------------
    # Compare against Binary ALS.
    # --------------------------------------------------------

    binary_results = sorted_results[
        sorted_results["playtime_beta"] == 0
    ]

    if not binary_results.empty:

        best_binary = binary_results.iloc[0]

        ndcg_difference = (
            best["ndcg_at_10"]
            - best_binary["ndcg_at_10"]
        ) * 100

        print("\nComparison with Binary ALS")

        print(
            f"Binary NDCG@10: "
            f"{best_binary['ndcg_at_10'] * 100:.2f}%"
        )

        print(
            f"Weighted NDCG@10: "
            f"{best['ndcg_at_10'] * 100:.2f}%"
        )

        print(
            f"NDCG difference: "
            f"{ndcg_difference:+.2f}%p"
        )

    print(f"\nFull results saved to: {RESULT_PATH}")


# ============================================================
# 8. Main execution
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load data.
    # --------------------------------------------------------

    train_df = pd.read_parquet(
        DATA_DIR / "steam_train.parquet",
        columns=[
            "user_id",
            "unified_game_id",
            "playtime_forever"
        ]
    )

    val_observed = pd.read_parquet(
        DATA_DIR / "steam_val_observed.parquet",
        columns=[
            "user_id",
            "unified_game_id"
        ]
    )

    val_heldout = pd.read_parquet(
        DATA_DIR / "steam_val_heldout.parquet",
        columns=[
            "user_id",
            "unified_game_id"
        ]
    )

    # --------------------------------------------------------
    # 2. Validate data.
    # --------------------------------------------------------

    validate_interactions(
        train_df,
        "Steam Train"
    )

    validate_interactions(
        val_observed,
        "Steam Validation Observed"
    )

    validate_interactions(
        val_heldout,
        "Steam Validation Heldout"
    )

    train_df = validate_train_playtime(
        train_df
    )

    print("=" * 60)
    print("Playtime Weighted ALS - Grid Search")
    print("=" * 60)

    print(f"Training interactions: {len(train_df):,}")
    print(f"Validation observed: {len(val_observed):,}")
    print(f"Validation held-out: {len(val_heldout):,}")

    # --------------------------------------------------------
    # 3. Prepare training data.
    # --------------------------------------------------------

    train_data = prepare_train_data(
        train_df
    )

    (
        _,
        _,
        playtime_hours,
        matrix_shape,
        game_ids,
        game_to_idx
    ) = train_data

    print(f"\nTraining matrix shape: {matrix_shape}")
    print(f"Candidate games: {len(game_ids):,}")

    # --------------------------------------------------------
    # 4. Calculate candidate caps.
    # --------------------------------------------------------

    cap_grid = calculate_cap_grid(
        playtime_hours
    )

    print("\nPlaytime Cap Candidates")

    for percentile, cap_hours in cap_grid:

        print(
            f"{percentile * 100:.0f}th percentile: "
            f"{cap_hours:.2f} hours"
        )

    # --------------------------------------------------------
    # 5. Build the validation fold-in matrix.
    # --------------------------------------------------------

    # Use ownership information only during evaluation.
    val_matrix, val_user_ids = build_validation_matrix(
        val_observed,
        game_ids,
        game_to_idx
    )

    # --------------------------------------------------------
    # 6. Grid Search
    # --------------------------------------------------------

    results_df = grid_search(
        train_data,
        val_matrix,
        val_user_ids,
        val_observed,
        val_heldout,
        cap_grid
    )

    # --------------------------------------------------------
    # 7. Display the best results.
    # --------------------------------------------------------

    print_best_result(
        results_df
    )


if __name__ == "__main__":
    main()
