
import os

# Prevent excessive OpenBLAS thread usage.
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

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

TOP_K = 10
BATCH_SIZE = 256

FACTORS = 24
REGULARIZATION = 5.0
ITERATIONS = 40
CONFIDENCE_ALPHA = 2
RANDOM_STATE = 42


# ============================================================
# 2. Training matrix construction
# ============================================================

def build_train_matrix(train_df):
    """
    Convert Steam training interactions into a user-by-game CSR matrix.

    Binary ALS:
        Owned games: confidence = 1 + CONFIDENCE_ALPHA (= 3).
        Unowned games: not stored in the sparse matrix.
    """

    # factorize() returns (codes, uniques). Each user ID gets a row code;
    # user_ids keeps the corresponding original IDs for reverse lookup.
    # sort=True orders the unique IDs and adjusts the codes accordingly.
    user_indices, user_ids = pd.factorize(
        train_df["user_id"],
        sort=True
    )

    # Apply the same encoding to game IDs for matrix column positions.
    # game_ids maps column positions back to original unified game IDs.
    game_indices, game_ids = pd.factorize(
        train_df["unified_game_id"],
        sort=True
    )

    # Assign confidence to owned games.
    confidence = np.full(
        len(train_df),
        1 + CONFIDENCE_ALPHA,
        dtype=np.float32
    )

    # CSR (Compressed Sparse Row) stores only observed interactions, avoiding
    # allocation of a dense user-by-game matrix. Each confidence[i] is placed
    # at row=user_indices[i], column=game_indices[i].
    # shape sets (number of users, number of games); dtype sets value precision.
    train_matrix = csr_matrix(
        (
            confidence,
            (user_indices, game_indices)
        ),
        shape=(len(user_ids), len(game_ids)),
        dtype=np.float32
    )

    # Map game IDs to matrix column indices.
    game_to_idx = {
        game_id: idx
        for idx, game_id in enumerate(game_ids)
    }

    return train_matrix, game_ids, game_to_idx


# ============================================================
# 3. Binary ALS training
# ============================================================

def train_als(train_matrix):
    """
    Train the Binary ALS model.
    """

    # Implicit ALS alternates between learning user and game latent factors.
    # API: https://benfred.github.io/implicit/api/models/cpu/als.html
    model = AlternatingLeastSquares(
        factors=FACTORS,                # Dimensions of each latent vector
        regularization=REGULARIZATION,  # L2 penalty to limit overfitting
        iterations=ITERATIONS,          # Number of ALS optimization passes
        alpha=1.0,                     # Do not rescale precomputed confidence
        random_state=RANDOM_STATE,     # Seed for factor initialization
        dtype=np.float32               # Store latent factors in 32-bit floats
    )

    print("\nStarting ALS training")
    # fit() factorizes the user-by-game CSR confidence matrix. Nonzero entries
    # denote owned games; missing entries imply preference=0, confidence=1.
    model.fit(train_matrix)
    print("ALS training completed")

    print(f"User factors: {model.user_factors.shape}")
    print(f"Game factors: {model.item_factors.shape}")

    return model


# ============================================================
# 4. Validation games grouped by user
# ============================================================

def group_games_by_user(df):
    """
    Group game IDs into a set for each user.
    """

    return (
        df.groupby("user_id")["unified_game_id"]
        .agg(set)
        .to_dict()
    )


# ============================================================
# 5. Batch user matrix construction
# ============================================================

def build_batch_matrix(
    batch_users,
    observed_by_user,
    game_to_idx,
    num_games
):
    """
    Convert validation users' observed games into a batch CSR matrix.

    Validation users are unseen during training, so a separate matrix
    is constructed for fold-in.
    """

    rows = []
    cols = []

    for local_user_idx, user_id in enumerate(batch_users):

        observed_games = observed_by_user[user_id]

        for game_id in observed_games:

            game_idx = game_to_idx.get(game_id)

            # Include only games present in the training set.
            if game_idx is not None:
                rows.append(local_user_idx)
                cols.append(game_idx)

    confidence = np.full(
        len(rows),
        1 + CONFIDENCE_ALPHA,
        dtype=np.float32
    )

    # Build a CSR matrix for unseen validation users with the same game columns
    # as the training matrix. rows contains batch-local user indices, cols
    # contains known training-game indices, and confidence holds their weights.
    # shape includes all batch users and training games, even if unobserved.
    batch_matrix = csr_matrix(
        (
            confidence,
            (rows, cols)
        ),
        shape=(len(batch_users), num_games),
        dtype=np.float32
    )

    return batch_matrix


# ============================================================
# 6. Batched fold-in recommendations
# ============================================================

def recommend_batch(
    model,
    batch_users,
    observed_by_user,
    game_ids,
    game_to_idx,
    k=10
):
    """
    Fold in multiple validation users and recommend their top-k games.

    Exclude previously observed games from recommendations.
    """

    # Build the batch matrix.
    batch_matrix = build_batch_matrix(
        batch_users,
        observed_by_user,
        game_to_idx,
        len(game_ids)
    )

    # Use local user indices within the batch.
    local_user_ids = np.arange(len(batch_users))

    # Fold in unseen users using Steam-trained game factors. With
    # recalculate_user=True, the model computes new user factors from the
    # observed games rather than using stored training-user factors.
    item_indices, scores = model.recommend(
        userid=local_user_ids,              # Indices of rows in batch_matrix
        user_items=batch_matrix,            # Observed-game confidence CSR
        N=k,                                # Number of ranked games per user
        filter_already_liked_items=True,   # Exclude observed games
        recalculate_user=True              # Derive factors for unseen users
    )
    # The result contains parallel arrays of game-column indices and scores.

    all_recommendations = []

    for row in range(len(batch_users)):

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

        all_recommendations.append(recommendations)

    return all_recommendations


# ============================================================
# 7. Per-user evaluation
# ============================================================

def evaluate_user(recommendations, heldout_games, k=10):
    """
    Evaluate the recommendations for one user.

    Metrics:
        Recall@10
        NDCG@10
        Hit@10
    """

    recommended_games = [
        game_id
        for game_id, _ in recommendations[:k]
    ]

    # Mark whether each recommended game is in the held-out set.
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
# 8. Full Steam validation evaluation
# ============================================================

def evaluate_all_users(
    model,
    val_observed,
    val_heldout,
    game_ids,
    game_to_idx,
    k=10,
    batch_size=256
):
    """
    Fold in and evaluate all validation users in batches.

    Return per-user metrics as a DataFrame for subsequent averaging.
    """

    observed_by_user = group_games_by_user(val_observed)
    heldout_by_user = group_games_by_user(val_heldout)

    # Ensure observed and held-out data contain the same users.
    if set(observed_by_user) != set(heldout_by_user):
        raise ValueError(
            "Observed and held-out user sets do not match."
        )

    user_ids = list(observed_by_user.keys())
    num_users = len(user_ids)

    results = []

    print("\nStarting full Steam validation evaluation")
    print(f"Evaluation users: {num_users:,}")
    print(f"Batch size: {batch_size}")

    # Evaluate users in batches.
    for start in range(0, num_users, batch_size):

        batch_users = user_ids[
            start:start + batch_size
        ]

        # Generate recommendations using fold-in.
        batch_recommendations = recommend_batch(
            model,
            batch_users,
            observed_by_user,
            game_ids,
            game_to_idx,
            k=k
        )

        # Compute per-user metrics.
        for user_id, recommendations in zip(
            batch_users,
            batch_recommendations
        ):

            observed_games = observed_by_user[user_id]
            heldout_games = heldout_by_user[user_id]

            recommended_games = [
                game_id
                for game_id, _ in recommendations
            ]

            # Check for overlap between observed and held-out games.
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

            # Ensure recommended games are unique.
            if len(recommended_games) != len(
                set(recommended_games)
            ):
                raise ValueError(
                    f"User {user_id}: duplicate recommendations"
                )

            # Evaluate the recommendations.
            metrics = evaluate_user(
                recommendations,
                heldout_games,
                k=k
            )

            results.append({
                "user_id": user_id,
                **metrics
            })

        # Display evaluation progress.
        completed = min(
            start + batch_size,
            num_users
        )

        print(
            f"\rEvaluation progress: {completed:,}/{num_users:,} users",
            end="",
            flush=True
        )

    print("\nFull user evaluation completed")

    return pd.DataFrame(results)


# ============================================================
# 9. Main execution
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

    print("=" * 55)
    print("Binary ALS - Steam Validation")
    print("=" * 55)

    print(f"Training interactions: {len(train_df):,}")
    print(f"Validation observed: {len(val_observed):,}")
    print(f"Validation held-out: {len(val_heldout):,}")

    # 2. Check for duplicate training interactions.
    if train_df.duplicated(
        ["user_id", "unified_game_id"]
    ).any():
        raise ValueError(
            "The training data contain duplicate user-game interactions."
        )

    # 3. Build the training matrix.
    train_matrix, game_ids, game_to_idx = (
        build_train_matrix(train_df)
    )

    print(f"\nTraining matrix shape: {train_matrix.shape}")
    print(f"Candidate games: {len(game_ids):,}")

    # 4. Train ALS.
    model = train_als(train_matrix)

    # 5. Evaluate on the full Steam validation split.
    results_df = evaluate_all_users(
        model,
        val_observed,
        val_heldout,
        game_ids,
        game_to_idx,
        k=TOP_K,
        batch_size=BATCH_SIZE
    )

    # 6. Average per-user metrics.
    mean_metrics = results_df[
        ["recall_at_10", "ndcg_at_10", "hit_at_10"]
    ].mean()

    # 7. Display results.
    print("\n" + "=" * 55)
    print("Binary ALS - Steam Validation Results")
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
