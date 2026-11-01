
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

TOP_K = 10
BATCH_SIZE = 256

FACTORS = 24
REGULARIZATION = 5.0
ITERATIONS = 40
RANDOM_STATE = 42

# Confidence 설정
OWNER_ALPHA = 2
PLAYTIME_BETA = 0.1

# 플레이타임 상한값: Train의 양수 플레이타임 75% 분위수
PLAYTIME_PERCENTILE = 0.75


# ============================================================
# 2. 데이터 검증
# ============================================================

def validate_interactions(df, name):
    """
    사용자-게임 관계의 결측값과 중복을 검사합니다.
    """

    required = {"user_id", "unified_game_id"}

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


def validate_train_playtime(train_df):
    """
    Train 데이터의 플레이타임을 검증합니다.
    """

    if "playtime_forever" not in train_df.columns:
        raise ValueError(
            "Steam Train에 playtime_forever 컬럼이 없습니다."
        )

    train_df = train_df.copy()

    train_df["playtime_forever"] = pd.to_numeric(
        train_df["playtime_forever"],
        errors="coerce"
    )

    values = train_df["playtime_forever"].to_numpy()

    if not np.isfinite(values).all():
        raise ValueError(
            "Steam Train에 유효하지 않은 플레이타임이 있습니다."
        )

    if (values < 0).any():
        raise ValueError(
            "Steam Train에 음수 플레이타임이 있습니다."
        )

    return train_df


# ============================================================
# 3. 플레이타임 상한값 계산
# ============================================================

def calculate_playtime_cap(train_df):
    """
    Steam Train에서 양수 플레이타임의
    99% 분위수를 계산합니다.

    Validation 및 Test 데이터는 사용하지 않습니다.
    """

    positive_minutes = train_df.loc[
        train_df["playtime_forever"] > 0,
        "playtime_forever"
    ]

    if positive_minutes.empty:
        raise ValueError(
            "학습 데이터에 양수 플레이타임이 없습니다."
        )

    # 분 → 시간
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
# 4. 플레이타임 기반 Confidence 계산
# ============================================================

def calculate_confidence(playtime_minutes, cap_hours):
    """
    소유 게임의 Confidence를 계산합니다.

    c_ui = 1 + OWNER_ALPHA
           + PLAYTIME_BETA * log(1 + min(t_ui, T))

    t_ui: 시간 단위 플레이타임
    T: Train에서 계산한 플레이타임 상한값

    플레이타임이 0분이어도 소유 정보는 유지합니다.
    """

    # 분 → 시간
    playtime_hours = (
        np.asarray(
            playtime_minutes,
            dtype=np.float64
        ) / 60
    )

    # 상한값 적용
    clipped_hours = np.minimum(
        playtime_hours,
        cap_hours
    )

    # 로그 변환
    playtime_weight = np.log1p(clipped_hours)

    # 최종 Confidence
    confidence = (
        1
        + OWNER_ALPHA
        + PLAYTIME_BETA * playtime_weight
    )

    return confidence.astype(np.float32)


# ============================================================
# 5. Steam Train 행렬 생성
# ============================================================

def build_train_matrix(train_df, cap_hours):
    """
    Steam Train 데이터를 사용자 × 게임 CSR 행렬로 변환합니다.

    소유 게임에는 플레이타임 기반 Confidence를 부여합니다.
    """

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

    # 플레이타임 기반 Confidence
    confidence = calculate_confidence(
        train_df["playtime_forever"].to_numpy(),
        cap_hours
    )

    # CSR 행렬 생성
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

    # 게임 ID → 행렬 인덱스
    game_to_idx = {
        game_id: idx
        for idx, game_id in enumerate(game_ids)
    }

    return train_matrix, game_ids, game_to_idx


# ============================================================
# 6. Weighted ALS 학습
# ============================================================

def train_als(train_matrix):
    """
    Playtime Weighted ALS 모델을 학습합니다.
    """

    model = AlternatingLeastSquares(
        factors=FACTORS,
        regularization=REGULARIZATION,
        iterations=ITERATIONS,
        alpha=1.0,
        random_state=RANDOM_STATE,
        dtype=np.float32
    )

    print("\nWeighted ALS 학습 시작")

    model.fit(train_matrix)

    print("Weighted ALS 학습 완료")

    print(
        f"사용자 잠재벡터: "
        f"{model.user_factors.shape}"
    )

    print(
        f"게임 잠재벡터: "
        f"{model.item_factors.shape}"
    )

    return model


# ============================================================
# 7. Steam Validation 행렬 생성
# ============================================================

def build_validation_matrix(
    val_observed,
    game_ids,
    game_to_idx
):
    """
    Steam Validation 사용자의 Fold-in 행렬을 생성합니다.

    평가 단계에서는 플레이타임을 사용하지 않습니다.

    관측된 소유 게임:
        confidence = 1 + OWNER_ALPHA

    관측되지 않은 게임:
        행렬에 저장하지 않음
    """

    # Validation 사용자 인덱싱
    user_indices, val_user_ids = pd.factorize(
        val_observed["user_id"],
        sort=True
    )

    # 학습 게임 인덱스로 변환
    mapped_game_indices = (
        val_observed["unified_game_id"]
        .map(game_to_idx)
    )

    # 학습 후보에 존재하는 게임만 사용
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

    # Validation 사용자 × 게임 행렬
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
        f"\nValidation Observed 중 "
        f"학습 후보에 없는 게임 기록: {unknown_count:,}건"
    )

    return val_matrix, val_user_ids


# ============================================================
# 8. 사용자별 평가
# ============================================================

def evaluate_user(recommended_games, heldout_games, k=10):
    """
    사용자 한 명의 Top-K 추천 성능을 평가합니다.

    Recall@K
    NDCG@K
    Hit@K
    """

    recommended_games = recommended_games[:k]

    # 추천 게임별 정답 여부
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
# 9. 전체 Steam Validation 평가
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
    모든 Validation 사용자에게 Fold-in을 적용하고
    Top-K 추천 성능을 평가합니다.
    """

    # 사용자별 Observed 게임 집합
    observed_by_user = (
        val_observed.groupby("user_id")["unified_game_id"]
        .agg(set)
        .to_dict()
    )

    # 사용자별 Heldout 게임 집합
    heldout_by_user = (
        val_heldout.groupby("user_id")["unified_game_id"]
        .agg(set)
        .to_dict()
    )

    # 사용자 집합 검사
    if set(observed_by_user) != set(heldout_by_user):
        raise ValueError(
            "Observed와 Heldout의 사용자 집합이 다릅니다."
        )

    num_users = len(val_user_ids)

    results = []

    print("\n" + "=" * 55)
    print("Playtime Weighted ALS - Steam Validation")
    print("=" * 55)

    print(f"평가 사용자 수: {num_users:,}명")
    print(f"배치 크기: {batch_size}")

    # 배치 단위 추천 및 평가
    for start in range(0, num_users, batch_size):

        end = min(
            start + batch_size,
            num_users
        )

        # 배치 사용자 행렬
        batch_matrix = val_matrix[start:end]

        # 배치 내부 사용자 인덱스
        local_user_ids = np.arange(
            end - start
        )

        # Fold-in 추천
        item_indices, scores = model.recommend(
            userid=local_user_ids,
            user_items=batch_matrix,
            N=k,
            filter_already_liked_items=True,
            recalculate_user=True
        )

        # 사용자별 평가
        for local_idx, user_id in enumerate(
            val_user_ids[start:end]
        ):

            observed_games = observed_by_user[user_id]
            heldout_games = heldout_by_user[user_id]

            # 추천 결과를 실제 게임 ID로 변환
            recommended_games = [
                game_ids[int(item_idx)]
                for item_idx in item_indices[local_idx]
            ]

            # Observed / Heldout 중복 검사
            if not observed_games.isdisjoint(heldout_games):
                raise ValueError(
                    f"사용자 {user_id}: Observed/Heldout 중복"
                )

            # Observed 게임 추천 여부 검사
            if not set(recommended_games).isdisjoint(
                observed_games
            ):
                raise ValueError(
                    f"사용자 {user_id}: Observed 게임이 추천됨"
                )

            # 추천 게임 중복 검사
            if len(recommended_games) != len(
                set(recommended_games)
            ):
                raise ValueError(
                    f"사용자 {user_id}: 추천 게임 중복"
                )

            # 평가
            metrics = evaluate_user(
                recommended_games,
                heldout_games,
                k=k
            )

            results.append({
                "user_id": user_id,
                **metrics
            })

        # 평가 진행 상황
        print(
            f"\r평가 진행: {end:,}/{num_users:,}명",
            end="",
            flush=True
        )

    print("\n전체 사용자 평가 완료")

    return pd.DataFrame(results)


# ============================================================
# 10. 메인 실행
# ============================================================

def main():

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

    # Validation에는 소유 정보만 존재합니다.
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
    # 2. 데이터 검증
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

    # Train 플레이타임만 검증
    train_df = validate_train_playtime(train_df)

    print("=" * 55)
    print("Playtime Weighted ALS")
    print("=" * 55)

    print(f"학습 데이터: {len(train_df):,}건")
    print(f"Validation Observed: {len(val_observed):,}건")
    print(f"Validation Heldout: {len(val_heldout):,}건")

    # --------------------------------------------------------
    # 3. 플레이타임 상한값 계산
    # --------------------------------------------------------

    cap_hours = calculate_playtime_cap(train_df)

    print("\n모델 설정")

    print(f"Factors: {FACTORS}")
    print(f"Regularization: {REGULARIZATION}")
    print(f"Iterations: {ITERATIONS}")
    print(f"Owner Alpha: {OWNER_ALPHA}")
    print(f"Playtime Beta: {PLAYTIME_BETA}")

    print(
        f"플레이타임 상한: "
        f"{cap_hours:.2f}시간"
    )

    # --------------------------------------------------------
    # 4. 학습 행렬 생성
    # --------------------------------------------------------

    train_matrix, game_ids, game_to_idx = (
        build_train_matrix(
            train_df,
            cap_hours
        )
    )

    print(
        f"\n학습 행렬 크기: "
        f"{train_matrix.shape}"
    )

    print(f"추천 후보 게임: {len(game_ids):,}개")

    # --------------------------------------------------------
    # 5. Weighted ALS 학습
    # --------------------------------------------------------

    model = train_als(train_matrix)

    # --------------------------------------------------------
    # 6. Validation Fold-in 행렬 생성
    # --------------------------------------------------------

    val_matrix, val_user_ids = build_validation_matrix(
        val_observed,
        game_ids,
        game_to_idx
    )

    # --------------------------------------------------------
    # 7. 전체 Steam Validation 평가
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
    # 8. 평균 평가 지표 계산
    # --------------------------------------------------------

    mean_metrics = results_df[
        [
            "recall_at_10",
            "ndcg_at_10",
            "hit_at_10"
        ]
    ].mean()

    # --------------------------------------------------------
    # 9. 결과 출력
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("Playtime Weighted ALS - Steam Validation 결과")
    print("=" * 55)

    print(f"평가 사용자 수: {len(results_df):,}명")

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

    print("\n백분율 기준")

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
