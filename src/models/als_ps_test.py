
import os

# OpenBLAS 스레드 설정
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from pathlib import Path
import math

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from implicit.cpu.als import AlternatingLeastSquares


# ============================================================
# 1. 설정
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULT_DIR = PROJECT_DIR / "data" / "results"

RESULT_PATH = RESULT_DIR / "als_ps_test_results.csv"

TOP_K = 10
BATCH_SIZE = 256
RANDOM_STATE = 42

# 최종 ALS 파라미터
FACTORS = 24
REGULARIZATION = 5.0
ITERATIONS = 40

OWNER_ALPHA = 2

# Weighted ALS 최종 설정
PLAYTIME_BETA = 0.1
PLAYTIME_PERCENTILE = 0.75


# ============================================================
# 2. 데이터 검증
# ============================================================

def validate_interactions(df, name):

    required = {
        "user_id",
        "unified_game_id"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{name}: 필수 컬럼 누락 {missing}"
        )

    if df[["user_id", "unified_game_id"]].isna().any().any():
        raise ValueError(
            f"{name}: 사용자 또는 게임 ID 결측값"
        )

    if df.duplicated(
        ["user_id", "unified_game_id"]
    ).any():
        raise ValueError(
            f"{name}: 사용자-게임 중복 기록"
        )


# ============================================================
# 3. Train 데이터 인덱싱
# ============================================================

def prepare_train_data(train_df):

    # 사용자 인덱싱
    user_indices, user_ids = pd.factorize(
        train_df["user_id"],
        sort=True
    )

    # 게임 인덱싱
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
# 4. ALS 학습 행렬 생성
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
                "Weighted ALS에는 cap_hours가 필요합니다."
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
# 5. ALS 학습
# ============================================================

def train_als(train_matrix, model_name):

    model = AlternatingLeastSquares(
        factors=FACTORS,
        regularization=REGULARIZATION,
        iterations=ITERATIONS,
        alpha=1.0,
        random_state=RANDOM_STATE,
        dtype=np.float32
    )

    print(f"\n{model_name} 학습 시작")

    model.fit(train_matrix)

    print(f"{model_name} 학습 완료")

    return model


# ============================================================
# 6. PlayStation Fold-in 행렬 생성
# ============================================================

def build_ps_matrix(
    ps_observed,
    game_ids,
    game_to_idx
):
    """
    PlayStation 사용자에게 관측된 게임을
    Binary Confidence로 변환합니다.

    PlayStation 플레이타임은 사용하지 않습니다.
    """

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
        f"학습 후보에 없는 PlayStation Observed 게임: "
        f"{unknown_count:,}건"
    )

    return ps_matrix, ps_user_ids


# ============================================================
# 7. 평가 지표
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
# 8. 전체 ALS 평가
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
    모든 PlayStation 사용자에게 Fold-in을 수행합니다.
    """

    num_users = len(ps_user_ids)

    results = []

    print("\n" + "=" * 55)
    print(f"{model_name} - PlayStation Test")
    print("=" * 55)

    print(f"평가 사용자 수: {num_users:,}명")

    for start in range(0, num_users, BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            num_users
        )

        batch_matrix = ps_matrix[start:end]

        local_user_ids = np.arange(
            end - start
        )

        # Fold-in Top-10 추천
        item_indices, scores = model.recommend(
            userid=local_user_ids,
            user_items=batch_matrix,
            N=TOP_K,
            filter_already_liked_items=True,
            recalculate_user=True
        )

        for local_idx, user_id in enumerate(
            ps_user_ids[start:end]
        ):

            observed_games = observed_by_user[user_id]
            heldout_games = heldout_by_user[user_id]

            recommended_games = [
                game_ids[int(item_idx)]
                for item_idx in item_indices[local_idx]
            ]

            # 추천 결과 검증
            if not set(recommended_games).isdisjoint(
                observed_games
            ):
                raise ValueError(
                    f"{user_id}: Observed 게임이 추천됨"
                )

            if len(recommended_games) != len(
                set(recommended_games)
            ):
                raise ValueError(
                    f"{user_id}: 추천 게임 중복"
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
            f"\r평가 진행: {end:,}/{num_users:,}명",
            end="",
            flush=True
        )

    print("\n평가 완료")

    return pd.DataFrame(results)


# ============================================================
# 9. Popularity 평가
# ============================================================

def evaluate_popularity(
    train_df,
    ps_user_ids,
    observed_by_user,
    heldout_by_user
):
    """
    Steam Train의 게임별 소유자 수를 기준으로
    Popularity Baseline을 평가합니다.
    """

    popularity = (
        train_df.groupby("unified_game_id")["user_id"]
        .nunique()
        .reset_index(name="owner_count")
        .sort_values(
            by=["owner_count", "unified_game_id"],
            ascending=[False, True]
        )
    )

    ranked_games = popularity[
        "unified_game_id"
    ].tolist()

    results = []

    print("\n" + "=" * 55)
    print("Popularity Baseline - PlayStation Test")
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
                f"\r평가 진행: {idx:,}/{len(ps_user_ids):,}명",
                end="",
                flush=True
            )

    print("\n평가 완료")

    return pd.DataFrame(results)


# ============================================================
# 10. 결과 요약
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
    print(f"{model_name} 평가 결과")
    print("=" * 55)

    print(f"평가 사용자 수: {len(results_df):,}명")

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
# 11. 메인 실행
# ============================================================

def main():

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. 데이터 불러오기
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
    # 2. 데이터 검증
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
            "Steam Train에 잘못된 플레이타임이 존재합니다."
        )

    print("=" * 55)
    print("Steam-to-PlayStation Final Evaluation")
    print("=" * 55)

    print(f"Steam Train: {len(train_df):,}건")
    print(f"PS Observed: {len(ps_observed):,}건")
    print(f"PS Heldout: {len(ps_heldout):,}건")

    # --------------------------------------------------------
    # 3. 사용자별 Observed / Heldout
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
            "PS Observed와 Heldout 사용자 집합이 다릅니다."
        )

    for user_id, observed_games in observed_by_user.items():

        if not observed_games.isdisjoint(
            heldout_by_user[user_id]
        ):
            raise ValueError(
                f"{user_id}: Observed와 Heldout 중복"
            )

    # --------------------------------------------------------
    # 4. 학습 데이터 인덱싱
    # --------------------------------------------------------

    (
        user_indices,
        game_indices,
        game_ids,
        game_to_idx,
        matrix_shape
    ) = prepare_train_data(train_df)

    print(f"\n학습 행렬 크기: {matrix_shape}")

    # --------------------------------------------------------
    # 5. PS Fold-in 행렬 생성
    # --------------------------------------------------------

    ps_matrix, ps_user_ids = build_ps_matrix(
        ps_observed,
        game_ids,
        game_to_idx
    )

    # --------------------------------------------------------
    # 6. Popularity Baseline 평가
    # --------------------------------------------------------

    popularity_results = evaluate_popularity(
        train_df,
        ps_user_ids,
        observed_by_user,
        heldout_by_user
    )

    summary_rows = [
        summarize_results(
            popularity_results,
            "Popularity"
        )
    ]

    # --------------------------------------------------------
    # 7. Binary ALS 학습 및 평가
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
    # 8. Weighted ALS 학습 및 평가
    # --------------------------------------------------------

    # Train의 양수 플레이타임에서 상한값 계산
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
        f"{cap_hours:.2f}시간"
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
    # 9. 전체 결과 요약
    # --------------------------------------------------------

    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 55)
    print("PlayStation Test - 최종 모델 비교")
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
    # 10. 결과 저장
    # --------------------------------------------------------

    summary_df.to_csv(
        RESULT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n결과 저장 완료: {RESULT_PATH}")


if __name__ == "__main__":
    main()
