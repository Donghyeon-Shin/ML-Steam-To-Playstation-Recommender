
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
RESULT_DIR = PROJECT_DIR / "data" / "results"

RESULT_PATH = RESULT_DIR / "als_ps_test_results.csv"

TOP_K = 10
BATCH_SIZE = 256
RANDOM_STATE = 42

# Final ALS hyperparameters
FACTORS = 24
REGULARIZATION = 5.0
ITERATIONS = 40

OWNER_ALPHA = 2

# Final Weighted ALS configuration
PLAYTIME_BETA = 0.1
PLAYTIME_PERCENTILE = 0.75


# ============================================================
# 2. Data validation
# ============================================================

def validate_interactions(df, name):

    required = {
        "user_id",
        "unified_game_id"
    }

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


# ============================================================
# 3. Training data indexing
# ============================================================

def prepare_train_data(train_df):

    # factorize() returns (codes, uniques): user_indices provides consecutive
    # matrix row numbers and user_ids stores their original Steam IDs.
    # sort=True sorts the unique IDs and remaps the codes consistently.
    user_indices, user_ids = pd.factorize(
        train_df["user_id"],
        sort=True
    )

    # Encode game IDs as column numbers; game_ids provides reverse lookup.
    game_indices, game_ids = pd.factorize(
        train_df["unified_game_id"],
        sort=True
    )

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
        game_ids,
        game_to_idx,
        matrix_shape
    )


# ============================================================
# 4. ALS training matrix construction
# ============================================================

def build_train_matrix(
    train_df,
    user_indices,
    game_indices,
    matrix_shape,
    weighted=False,
    cap_hours=None
):
    """
    Binary:
        c_ui = 1 + OWNER_ALPHA

    Weighted:
        c_ui = 1 + OWNER_ALPHA
               + PLAYTIME_BETA * log(1 + min(t_ui, T))
    """

    if weighted:

        if cap_hours is None:
            raise ValueError(
                "Weighted ALS requires cap_hours."
            )

        playtime_hours = (
            train_df["playtime_forever"]
            .to_numpy(dtype=np.float64) / 60
        )

        clipped_hours = np.minimum(
            playtime_hours,
            cap_hours
        )

        confidence = (
            1
            + OWNER_ALPHA
            + PLAYTIME_BETA * np.log1p(clipped_hours)
        ).astype(np.float32)

    else:

        confidence = np.full(
            len(train_df),
            1 + OWNER_ALPHA,
            dtype=np.float32
        )

    # Build a CSR (Compressed Sparse Row) confidence matrix without storing
    # all unobserved user-game pairs. confidence[i] is placed at the pair
    # (user_indices[i], game_indices[i]); matrix_shape fixes the full size
    # and dtype=np.float32 controls stored value precision.
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
# 5. ALS training
# ============================================================

def train_als(train_matrix, model_name):

    # Implicit CPU ALS alternates optimization of latent user/game vectors.
    # API: https://benfred.github.io/implicit/api/models/cpu/als.html
    model = AlternatingLeastSquares(
        factors=FACTORS,                # Dimensions of latent vectors
        regularization=REGULARIZATION,  # L2 penalty against overfitting
        iterations=ITERATIONS,          # ALS optimization passes
        alpha=1.0,                     # Preserve precomputed confidence
        random_state=RANDOM_STATE,     # Seed factor initialization
        dtype=np.float32               # Use 32-bit factor vectors
    )

    print(f"\nStarting {model_name} training")

    # fit() learns factors from Steam's user-by-game confidence CSR. Missing
    # interactions are unobserved with baseline confidence 1.
    model.fit(train_matrix)

    print(f"{model_name} training completed")

    return model


# ============================================================
# 6. PlayStation fold-in matrix construction
# ============================================================

def build_ps_matrix(
    ps_observed,
    game_ids,
    game_to_idx
):
    """
    Convert PlayStation users' observed games to binary confidence.

    PlayStation playtimes are not used.
    """

    # factorize() maps PS users to consecutive CSR row indices; ps_user_ids
    # holds the sorted original PS IDs used to align evaluation results.
    user_indices, ps_user_ids = pd.factorize(
        ps_observed["user_id"],
        sort=True
    )

    mapped_game_indices = (
        ps_observed["unified_game_id"]
        .map(game_to_idx)
    )

    known_mask = mapped_game_indices.notna().to_numpy()

    known_user_indices = user_indices[known_mask]

    known_game_indices = (
        mapped_game_indices[known_mask]
        .to_numpy(dtype=np.int32)
    )

    confidence = np.full(
        len(known_user_indices),
        1 + OWNER_ALPHA,
        dtype=np.float32
    )

    # Construct PS users' sparse fold-in matrix with Steam game columns.
    # The coordinate arrays identify each known user-game interaction and
    # confidence provides its weight. The full shape includes all PS users
    # and Steam candidate games, with absent pairs implicitly zero.
    ps_matrix = csr_matrix(
        (
            confidence,
            (
                known_user_indices,
                known_game_indices
            )
        ),
        shape=(
            len(ps_user_ids),
            len(game_ids)
        ),
        dtype=np.float32
    )

    unknown_count = (~known_mask).sum()

    print(
        f"PlayStation observed interactions with games "
        f"outside the training candidates: {unknown_count:,}"
    )

    return ps_matrix, ps_user_ids


# ============================================================
# 7. Evaluation metrics
# ============================================================

def evaluate_user(recommended_games, heldout_games, k=10):

    recommended_games = recommended_games[:k]

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

    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "recall_at_10": recall,
        "ndcg_at_10": ndcg,
        "hit_at_10": hit
    }


# ============================================================
# 8. Full ALS evaluation
# ============================================================

def evaluate_als(
    model,
    ps_matrix,
    ps_user_ids,
    observed_by_user,
    heldout_by_user,
    game_ids,
    model_name
):
    """
    Fold in and evaluate all PlayStation users.
    """

    num_users = len(ps_user_ids)

    results = []

    print("\n" + "=" * 55)
    print(f"{model_name} - PlayStation Test")
    print("=" * 55)

    print(f"Evaluation users: {num_users:,}")

    for start in range(0, num_users, BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            num_users
        )

        batch_matrix = ps_matrix[start:end]

        local_user_ids = np.arange(
            end - start
        )

        # Use Steam-trained game factors for PlayStation users. Fold-in
        # recalculates their user factors using only overlapping observed games.
        item_indices, scores = model.recommend(
            userid=local_user_ids,              # Batch-local CSR row indices
            user_items=batch_matrix,            # PS observed-game confidence
            N=TOP_K,                            # Number of games per PS user
            filter_already_liked_items=True,   # Exclude observed games
            recalculate_user=True              # Fold in unseen PS users
        )
        # Parallel arrays: game-column indices and recommendation scores.

        for local_idx, user_id in enumerate(
            ps_user_ids[start:end]
        ):

            observed_games = observed_by_user[user_id]
            heldout_games = heldout_by_user[user_id]

            recommended_games = [
                game_ids[int(item_idx)]
                for item_idx in item_indices[local_idx]
            ]

            # Validate the recommendations.
            if not set(recommended_games).isdisjoint(
                observed_games
            ):
                raise ValueError(
                    f"{user_id}: an observed game was recommended"
                )

            if len(recommended_games) != len(
                set(recommended_games)
            ):
                raise ValueError(
                    f"{user_id}: duplicate recommendations"
                )

            metrics = evaluate_user(
                recommended_games,
                heldout_games,
                k=TOP_K
            )

            results.append({
                "user_id": user_id,
                **metrics
            })

        print(
            f"\rEvaluation progress: {end:,}/{num_users:,} users",
            end="",
            flush=True
        )

    print("\nEvaluation completed")

    return pd.DataFrame(results)


# ============================================================
# 9. Popularity evaluation
# ============================================================

def rank_games_by_popularity(interactions_df, candidate_game_ids):
    """
    Rank the same Steam candidate catalog by distinct owner counts.

    Games absent from the popularity source receive zero owners and remain
    available as candidates. Ties are broken by unified game ID.
    """

    owner_counts = (
        interactions_df.groupby("unified_game_id")["user_id"]
        .nunique()
        .reset_index(name="owner_count")
    )

    popularity = (
        pd.DataFrame({"unified_game_id": candidate_game_ids})
        .merge(owner_counts, on="unified_game_id", how="left")
        .fillna({"owner_count": 0})
        .sort_values(
            by=["owner_count", "unified_game_id"],
            ascending=[False, True]
        )
    )

    return popularity["unified_game_id"].tolist()


def evaluate_popularity(
    popularity_source,
    candidate_game_ids,
    ps_user_ids,
    observed_by_user,
    heldout_by_user,
    model_name
):
    """
    Evaluate popularity-based recommendations on the same PS held-out split.

    Steam Popularity uses Steam Train ownership. PS Popularity uses only
    ps_test_observed: this is a target-adapted (transductive) baseline,
    not a strictly zero-shot baseline. PS held-out interactions are never
    used to calculate popularity.
    """

    ranked_games = rank_games_by_popularity(
        popularity_source,
        candidate_game_ids
    )

    results = []

    print("\n" + "=" * 55)
    print(f"{model_name} - PlayStation Test")
    print("=" * 55)

    for idx, user_id in enumerate(ps_user_ids, start=1):

        observed_games = observed_by_user[user_id]
        heldout_games = heldout_by_user[user_id]

        recommendations = []

        for game_id in ranked_games:

            if game_id not in observed_games:
                recommendations.append(game_id)

            if len(recommendations) == TOP_K:
                break

        metrics = evaluate_user(
            recommendations,
            heldout_games,
            k=TOP_K
        )

        results.append({
            "user_id": user_id,
            **metrics
        })

        if idx % 1000 == 0 or idx == len(ps_user_ids):

            print(
                f"\rEvaluation progress: {idx:,}/{len(ps_user_ids):,} users",
                end="",
                flush=True
            )

    print("\nEvaluation completed")

    return pd.DataFrame(results)


# ============================================================
# 10. Results summary
# ============================================================

def summarize_results(results_df, model_name):

    metrics = results_df[
        [
            "recall_at_10",
            "ndcg_at_10",
            "hit_at_10"
        ]
    ].mean()

    print("\n" + "=" * 55)
    print(f"{model_name} Evaluation Results")
    print("=" * 55)

    print(f"Evaluation users: {len(results_df):,}")

    print(
        f"Recall@10: "
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

    return {
        "model": model_name,
        "users": len(results_df),
        "recall_at_10": metrics["recall_at_10"],
        "ndcg_at_10": metrics["ndcg_at_10"],
        "hit_at_10": metrics["hit_at_10"]
    }


# ============================================================
# 11. Main execution
# ============================================================

def main():

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

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

    ps_observed = pd.read_parquet(
        DATA_DIR / "ps_test_observed.parquet",
        columns=[
            "user_id",
            "unified_game_id"
        ]
    )

    ps_heldout = pd.read_parquet(
        DATA_DIR / "ps_test_heldout.parquet",
        columns=[
            "user_id",
            "unified_game_id"
        ]
    )

    # --------------------------------------------------------
    # 2. Validate data.
    # --------------------------------------------------------

    validate_interactions(train_df, "Steam Train")
    validate_interactions(ps_observed, "PS Observed")
    validate_interactions(ps_heldout, "PS Heldout")

    train_df["playtime_forever"] = pd.to_numeric(
        train_df["playtime_forever"],
        errors="coerce"
    )

    if (
        not np.isfinite(
            train_df["playtime_forever"].to_numpy()
        ).all()
        or (train_df["playtime_forever"] < 0).any()
    ):
        raise ValueError(
            "Steam Train contains invalid playtime values."
        )

    print("=" * 55)
    print("Steam-to-PlayStation Final Evaluation")
    print("=" * 55)

    print(f"Steam Train: {len(train_df):,} interactions")
    print(f"PS Observed: {len(ps_observed):,} interactions")
    print(f"PS Heldout: {len(ps_heldout):,} interactions")

    # --------------------------------------------------------
    # 3. Group observed and held-out games by user.
    # --------------------------------------------------------

    observed_by_user = (
        ps_observed.groupby("user_id")["unified_game_id"]
        .agg(set)
        .to_dict()
    )

    heldout_by_user = (
        ps_heldout.groupby("user_id")["unified_game_id"]
        .agg(set)
        .to_dict()
    )

    if set(observed_by_user) != set(heldout_by_user):
        raise ValueError(
            "PS observed and held-out user sets do not match."
        )

    for user_id, observed_games in observed_by_user.items():

        if not observed_games.isdisjoint(
            heldout_by_user[user_id]
        ):
            raise ValueError(
                f"{user_id}: overlapping observed and held-out games"
            )

    # --------------------------------------------------------
    # 4. Index training data.
    # --------------------------------------------------------

    (
        user_indices,
        game_indices,
        game_ids,
        game_to_idx,
        matrix_shape
    ) = prepare_train_data(train_df)

    print(f"\nTraining matrix shape: {matrix_shape}")

    # --------------------------------------------------------
    # 5. Build the PS fold-in matrix.
    # --------------------------------------------------------

    ps_matrix, ps_user_ids = build_ps_matrix(
        ps_observed,
        game_ids,
        game_to_idx
    )

    # --------------------------------------------------------
    # 6. Evaluate both source-domain and target-adapted popularity.
    # --------------------------------------------------------

    steam_popularity_results = evaluate_popularity(
        train_df,
        game_ids,
        ps_user_ids,
        observed_by_user,
        heldout_by_user,
        "Steam Popularity"
    )

    summary_rows = [
        summarize_results(
            steam_popularity_results,
            "Steam Popularity"
        )
    ]

    # Use PS observed interactions only, never PS held-out ownership.
    ps_popularity_results = evaluate_popularity(
        ps_observed,
        game_ids,
        ps_user_ids,
        observed_by_user,
        heldout_by_user,
        "PS Popularity (Observed)"
    )

    summary_rows.append(
        summarize_results(
            ps_popularity_results,
            "PS Popularity (Observed)"
        )
    )

    # --------------------------------------------------------
    # 7. Train and evaluate Binary ALS.
    # --------------------------------------------------------

    binary_matrix = build_train_matrix(
        train_df,
        user_indices,
        game_indices,
        matrix_shape,
        weighted=False
    )

    binary_model = train_als(
        binary_matrix,
        "Binary ALS"
    )

    binary_results = evaluate_als(
        binary_model,
        ps_matrix,
        ps_user_ids,
        observed_by_user,
        heldout_by_user,
        game_ids,
        "Binary ALS"
    )

    summary_rows.append(
        summarize_results(
            binary_results,
            "Binary ALS"
        )
    )

    # --------------------------------------------------------
    # 8. Train and evaluate Weighted ALS.
    # --------------------------------------------------------

    # Calculate the cap from positive training playtimes.
    positive_hours = (
        train_df.loc[
            train_df["playtime_forever"] > 0,
            "playtime_forever"
        ].to_numpy(dtype=np.float64) / 60
    )

    cap_hours = float(
        np.quantile(
            positive_hours,
            PLAYTIME_PERCENTILE
        )
    )

    print(
        f"\nWeighted ALS Playtime Cap: "
        f"{cap_hours:.2f} hours"
    )

    weighted_matrix = build_train_matrix(
        train_df,
        user_indices,
        game_indices,
        matrix_shape,
        weighted=True,
        cap_hours=cap_hours
    )

    weighted_model = train_als(
        weighted_matrix,
        "Playtime Weighted ALS"
    )

    weighted_results = evaluate_als(
        weighted_model,
        ps_matrix,
        ps_user_ids,
        observed_by_user,
        heldout_by_user,
        game_ids,
        "Playtime Weighted ALS"
    )

    summary_rows.append(
        summarize_results(
            weighted_results,
            "Playtime Weighted ALS"
        )
    )

    # --------------------------------------------------------
    # 9. Summarize all results.
    # --------------------------------------------------------

    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 55)
    print("PlayStation Test - Final Model Comparison")
    print("=" * 55)

    display_df = summary_df.copy()

    for column in [
        "recall_at_10",
        "ndcg_at_10",
        "hit_at_10"
    ]:
        display_df[column] *= 100

    print(
        display_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # --------------------------------------------------------
    # 10. Save results.
    # --------------------------------------------------------

    summary_df.to_csv(
        RESULT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nResults saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
