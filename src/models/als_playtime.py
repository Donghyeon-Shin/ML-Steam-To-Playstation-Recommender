
import os

# Limit OpenBLAS to one thread.
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from pathlib import Path
import math

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from implicit.cpu.als import AlternatingLeastSquares


# ============================================================
# 1. Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"

TOP_K = 10
BATCH_SIZE = 256

FACTORS = 24
REGULARIZATION = 5.0
ITERATIONS = 40
RANDOM_STATE = 42

# Confidence parameters
OWNER_ALPHA = 2
PLAYTIME_BETA = 0.1

# Playtime cap: 75th percentile of positive training playtimes.
PLAYTIME_PERCENTILE = 0.75


# ============================================================
# 2. Data validation
# ============================================================

def validate_interactions(df, name):
    """
    Check user-game interactions for missing values and duplicates.
    """

    required = {"user_id", "unified_game_id"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{name}: missing required columns {missing}"
        )

    if df[["user_id", "unified_game_id"]].isna().any().any():
        raise ValueError(
            f"{name}: missing user or game IDs"
        )

    if df.duplicated(
        ["user_id", "unified_game_id"]
    ).any():
        raise ValueError(
            f"{name}: duplicate user-game interactions"
        )


def validate_train_playtime(train_df):
    """
    Validate playtime values in the training data.
    """

    if "playtime_forever" not in train_df.columns:
        raise ValueError(
            "Steam Train is missing the playtime_forever column."
        )

    train_df = train_df.copy()

    train_df["playtime_forever"] = pd.to_numeric(
        train_df["playtime_forever"],
        errors="coerce"
    )

    values = train_df["playtime_forever"].to_numpy()

    if not np.isfinite(values).all():
        raise ValueError(
            "Steam Train contains invalid playtime values."
        )

    if (values < 0).any():
        raise ValueError(
            "Steam Train contains negative playtime values."
        )

    return train_df


# ============================================================
# 3. Playtime cap calculation
# ============================================================

def calculate_playtime_cap(train_df):
    """
    Calculate the configured percentile of positive Steam Train playtimes.

    Validation and test data are excluded.
    """

    positive_minutes = train_df.loc[
        train_df["playtime_forever"] > 0,
        "playtime_forever"
    ]

    if positive_minutes.empty:
        raise ValueError(
            "The training data contain no positive playtime values."
        )

    # Convert minutes to hours.
    positive_hours = (
        positive_minutes.to_numpy(dtype=np.float64) / 60
    )

    cap_hours = float(
        np.quantile(
            positive_hours,
            PLAYTIME_PERCENTILE
        )
    )

    return cap_hours


# ============================================================
# 4. Playtime-weighted confidence calculation
# ============================================================

def calculate_confidence(playtime_minutes, cap_hours):
    """
    Compute confidence scores for owned games.

    c_ui = 1 + OWNER_ALPHA
           + PLAYTIME_BETA * log(1 + min(t_ui, T))

    t_ui: playtime measured in hours
    T: playtime cap calculated from the training data

    Ownership confidence is retained even when playtime is zero.
    """

    # Convert minutes to hours.
    playtime_hours = (
        np.asarray(
            playtime_minutes,
            dtype=np.float64
        ) / 60
    )

    # Cap the playtime values.
    clipped_hours = np.minimum(
        playtime_hours,
        cap_hours
    )

    # Apply a logarithmic transform.
    playtime_weight = np.log1p(clipped_hours)

    # Compute final confidence scores.
    confidence = (
        1
        + OWNER_ALPHA
        + PLAYTIME_BETA * playtime_weight
    )

    return confidence.astype(np.float32)


# ============================================================
# 5. Steam training matrix construction
# ============================================================

def build_train_matrix(train_df, cap_hours):
    """
    Convert Steam training interactions into a user-by-game CSR matrix.

    Weight owned games by their playtime-based confidence.
    """

    # factorize() returns (codes, uniques): user_indices are consecutive row
    # positions, while user_ids stores their original IDs for reverse lookup.
    # sort=True sorts distinct IDs and updates the codes consistently.
    user_indices, user_ids = pd.factorize(
        train_df["user_id"],
        sort=True
    )

    # Encode games as matrix column positions in the same way.
    # game_ids stores the original ID corresponding to each column.
    game_indices, game_ids = pd.factorize(
        train_df["unified_game_id"],
        sort=True
    )

    # Compute playtime-based confidence.
    confidence = calculate_confidence(
        train_df["playtime_forever"].to_numpy(),
        cap_hours
    )

    # CSR stores nonzero user-game interactions without allocating a dense
    # matrix. confidence[i] goes to (user_indices[i], game_indices[i]).
    # shape specifies user and game counts; dtype specifies stored precision.
    # The values here include both ownership and playtime-based confidence.
    train_matrix = csr_matrix(
        (
            confidence,
            (user_indices, game_indices)
        ),
        shape=(
            len(user_ids),
            len(game_ids)
        ),
        dtype=np.float32
    )

    # Map game IDs to matrix column indices.
    game_to_idx = {
        game_id: idx
        for idx, game_id in enumerate(game_ids)
    }

    return train_matrix, game_ids, game_to_idx


# ============================================================
# 6. Weighted ALS training
# ============================================================

def train_als(train_matrix):
    """
    Train the Playtime Weighted ALS model.
    """

    # Implicit ALS learns latent user/game factors from confidence-weighted
    # interactions, alternating the optimization of each factor matrix.
    # API: https://benfred.github.io/implicit/api/models/cpu/als.html
    model = AlternatingLeastSquares(
        factors=FACTORS,                # Dimensions of each latent vector
        regularization=REGULARIZATION,  # L2 penalty to limit overfitting
        iterations=ITERATIONS,          # Number of ALS optimization passes
        alpha=1.0,                     # Keep precomputed confidence unchanged
        random_state=RANDOM_STATE,     # Seed for factor initialization
        dtype=np.float32               # Use 32-bit latent factors
    )

    print("\nStarting Weighted ALS training")

    # fit() factorizes the user-by-game CSR matrix. Each nonzero entry stores
    # confidence for an owned game; missing entries use baseline confidence 1.
    model.fit(train_matrix)

    print("Weighted ALS training completed")

    print(
        f"User factors: "
        f"{model.user_factors.shape}"
    )

    print(
        f"Game factors: "
        f"{model.item_factors.shape}"
    )

    return model


# ============================================================
# 7. Steam validation matrix construction
# ============================================================

def build_validation_matrix(
    val_observed,
    game_ids,
    game_to_idx
):
    """
    Build the fold-in matrix for Steam validation users.

    Playtime is not used during evaluation.

    Observed owned games:
        confidence = 1 + OWNER_ALPHA

    Unobserved games:
        Not stored in the sparse matrix.
    """

    # factorize() maps each observed validation user ID to a consecutive row
    # code; val_user_ids stores the sorted original IDs (sort=True).
    # This row order is also used for batched fold-in evaluation.
    user_indices, val_user_ids = pd.factorize(
        val_observed["user_id"],
        sort=True
    )

    # Map games to their training-set indices.
    mapped_game_indices = (
        val_observed["unified_game_id"]
        .map(game_to_idx)
    )

    # Keep only games present in the training candidates.
    known_mask = mapped_game_indices.notna().to_numpy()

    known_user_indices = user_indices[known_mask]

    known_game_indices = (
        mapped_game_indices[known_mask]
        .to_numpy(dtype=np.int32)
    )

    # Binary Ownership Confidence
    confidence = np.full(
        len(known_user_indices),
        1 + OWNER_ALPHA,
        dtype=np.float32
    )

    # Construct a CSR matrix whose rows are validation users and whose
    # columns match the training games. The coordinate arrays select only
    # known user-game interactions; confidence supplies their binary weights.
    # shape retains all validation users and game columns, including zeros.
    val_matrix = csr_matrix(
        (
            confidence,
            (
                known_user_indices,
                known_game_indices
            )
        ),
        shape=(
            len(val_user_ids),
            len(game_ids)
        ),
        dtype=np.float32
    )

    unknown_count = (~known_mask).sum()

    print(
        f"\nValidation observed interactions with games "
        f"outside the training candidates: {unknown_count:,}"
    )

    return val_matrix, val_user_ids


# ============================================================
# 8. Per-user evaluation
# ============================================================

def evaluate_user(recommended_games, heldout_games, k=10):
    """
    Evaluate top-k recommendation quality for one user.

    Recall@K
    NDCG@K
    Hit@K
    """

    recommended_games = recommended_games[:k]

    # Mark held-out hits among recommended games.
    hits = [
        1 if game_id in heldout_games else 0
        for game_id in recommended_games
    ]

    correct_count = sum(hits)

    # Recall@K
    recall = (
        correct_count / len(heldout_games)
        if heldout_games else 0.0
    )

    # Hit@K
    hit = 1.0 if correct_count > 0 else 0.0

    # DCG@K
    dcg = sum(
        relevance / math.log2(rank + 2)
        for rank, relevance in enumerate(hits)
    )

    # IDCG@K
    ideal_hits = min(k, len(heldout_games))

    idcg = sum(
        1 / math.log2(rank + 2)
        for rank in range(ideal_hits)
    )

    # NDCG@K
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "recall_at_10": recall,
        "ndcg_at_10": ndcg,
        "hit_at_10": hit
    }


# ============================================================
# 9. Full Steam validation evaluation
# ============================================================

def evaluate_all_users(
    model,
    val_matrix,
    val_user_ids,
    val_observed,
    val_heldout,
    game_ids,
    k=10,
    batch_size=256
):
    """
    Fold in all validation users and evaluate top-k recommendations.
    """

    # Group observed games by user.
    observed_by_user = (
        val_observed.groupby("user_id")["unified_game_id"]
        .agg(set)
        .to_dict()
    )

    # Group held-out games by user.
    heldout_by_user = (
        val_heldout.groupby("user_id")["unified_game_id"]
        .agg(set)
        .to_dict()
    )

    # Verify that both splits contain the same users.
    if set(observed_by_user) != set(heldout_by_user):
        raise ValueError(
            "Observed and held-out user sets do not match."
        )

    num_users = len(val_user_ids)

    results = []

    print("\n" + "=" * 55)
    print("Playtime Weighted ALS - Steam Validation")
    print("=" * 55)

    print(f"Evaluation users: {num_users:,}")
    print(f"Batch size: {batch_size}")

    # Recommend and evaluate in batches.
    for start in range(0, num_users, batch_size):

        end = min(
            start + batch_size,
            num_users
        )

        # Slice the batch user matrix.
        batch_matrix = val_matrix[start:end]

        # Use local indices within the batch.
        local_user_ids = np.arange(
            end - start
        )

        # Fold in validation users using Steam-trained game factors.
        # recalculate_user=True derives each new user's latent factors from
        # observed ownership, without requiring the user in the training set.
        item_indices, scores = model.recommend(
            userid=local_user_ids,              # Batch-local CSR row indices
            user_items=batch_matrix,            # Observed ownership CSR
            N=k,                                # Number of games per user
            filter_already_liked_items=True,   # Exclude observed games
            recalculate_user=True              # Recompute new user factors
        )
        # item_indices and scores contain aligned rankings for each batch row.

        # Evaluate each user.
        for local_idx, user_id in enumerate(
            val_user_ids[start:end]
        ):

            observed_games = observed_by_user[user_id]
            heldout_games = heldout_by_user[user_id]

            # Convert recommendation indices to game IDs.
            recommended_games = [
                game_ids[int(item_idx)]
                for item_idx in item_indices[local_idx]
            ]

            # Check for observed/held-out overlap.
            if not observed_games.isdisjoint(heldout_games):
                raise ValueError(
                    f"User {user_id}: overlapping observed and held-out games"
                )

            # Ensure observed games were not recommended.
            if not set(recommended_games).isdisjoint(
                observed_games
            ):
                raise ValueError(
                    f"User {user_id}: an observed game was recommended"
                )

            # Check for duplicate recommendations.
            if len(recommended_games) != len(
                set(recommended_games)
            ):
                raise ValueError(
                    f"User {user_id}: duplicate recommendations"
                )

            # Evaluate recommendations.
            metrics = evaluate_user(
                recommended_games,
                heldout_games,
                k=k
            )

            results.append({
                "user_id": user_id,
                **metrics
            })

        # Display evaluation progress.
        print(
            f"\rEvaluation progress: {end:,}/{num_users:,} users",
            end="",
            flush=True
        )

    print("\nFull user evaluation completed")

    return pd.DataFrame(results)


# ============================================================
# 10. Main execution
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

    # Validation data contain ownership information only.
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

    # Validate playtime in training data only.
    train_df = validate_train_playtime(train_df)

    print("=" * 55)
    print("Playtime Weighted ALS")
    print("=" * 55)

    print(f"Training interactions: {len(train_df):,}")
    print(f"Validation observed: {len(val_observed):,}")
    print(f"Validation held-out: {len(val_heldout):,}")

    # --------------------------------------------------------
    # 3. Calculate the playtime cap.
    # --------------------------------------------------------

    cap_hours = calculate_playtime_cap(train_df)

    print("\nModel configuration")

    print(f"Factors: {FACTORS}")
    print(f"Regularization: {REGULARIZATION}")
    print(f"Iterations: {ITERATIONS}")
    print(f"Owner Alpha: {OWNER_ALPHA}")
    print(f"Playtime Beta: {PLAYTIME_BETA}")

    print(
        f"Playtime cap: "
        f"{cap_hours:.2f} hours"
    )

    # --------------------------------------------------------
    # 4. Build the training matrix.
    # --------------------------------------------------------

    train_matrix, game_ids, game_to_idx = (
        build_train_matrix(
            train_df,
            cap_hours
        )
    )

    print(
        f"\nTraining matrix shape: "
        f"{train_matrix.shape}"
    )

    print(f"Candidate games: {len(game_ids):,}")

    # --------------------------------------------------------
    # 5. Train Weighted ALS.
    # --------------------------------------------------------

    model = train_als(train_matrix)

    # --------------------------------------------------------
    # 6. Build the validation fold-in matrix.
    # --------------------------------------------------------

    val_matrix, val_user_ids = build_validation_matrix(
        val_observed,
        game_ids,
        game_to_idx
    )

    # --------------------------------------------------------
    # 7. Evaluate on the full Steam validation split.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 8. Average the evaluation metrics.
    # --------------------------------------------------------

    mean_metrics = results_df[
        [
            "recall_at_10",
            "ndcg_at_10",
            "hit_at_10"
        ]
    ].mean()

    # --------------------------------------------------------
    # 9. Display results.
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("Playtime Weighted ALS - Steam Validation Results")
    print("=" * 55)

    print(f"Evaluation users: {len(results_df):,}")

    print(
        f"\nRecall@10: "
        f"{mean_metrics['recall_at_10']:.4f}"
    )

    print(
        f"NDCG@10: "
        f"{mean_metrics['ndcg_at_10']:.4f}"
    )

    print(
        f"Hit Rate@10: "
        f"{mean_metrics['hit_at_10']:.4f}"
    )

    print("\nPercentage values")

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


if __name__ == "__main__":
    main()
