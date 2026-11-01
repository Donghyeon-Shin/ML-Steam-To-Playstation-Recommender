
import os

# OpenBLAS 과도한 스레드 사용 방지
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

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

TOP_K = 10
BATCH_SIZE = 256

FACTORS = 24
REGULARIZATION = 5.0
ITERATIONS = 40
CONFIDENCE_ALPHA = 2
RANDOM_STATE = 42


# ============================================================
# 2. 학습 데이터 행렬 생성
# ============================================================

def build_train_matrix(train_df):
    """
    Steam Train 데이터를 사용자 × 게임 CSR 행렬로 변환합니다.

    Binary ALS:
        소유한 게임: confidence = 1 + 20 = 21
        소유하지 않은 게임: 행렬에 저장하지 않음
    """

    # 사용자 및 게임 ID를 연속된 정수 인덱스로 변환
    user_indices, user_ids = pd.factorize(
        train_df["user_id"],
        sort=True
    )

    game_indices, game_ids = pd.factorize(
        train_df["unified_game_id"],
        sort=True
    )

    # 소유한 게임의 confidence
    confidence = np.full(
        len(train_df),
        1 + CONFIDENCE_ALPHA,
        dtype=np.float32
    )

    # 사용자 × 게임 희소 행렬
    train_matrix = csr_matrix(
        (
            confidence,
            (user_indices, game_indices)
        ),
        shape=(len(user_ids), len(game_ids)),
        dtype=np.float32
    )

    # 게임 ID → 행렬 인덱스
    game_to_idx = {
        game_id: idx
        for idx, game_id in enumerate(game_ids)
    }

    return train_matrix, game_ids, game_to_idx


# ============================================================
# 3. Binary ALS 학습
# ============================================================

def train_als(train_matrix):
    """
    Binary ALS 모델을 학습합니다.
    """

    model = AlternatingLeastSquares(
        factors=FACTORS,
        regularization=REGULARIZATION,
        iterations=ITERATIONS,
        alpha=1.0,
        random_state=RANDOM_STATE,
        dtype=np.float32
    )

    print("\nALS 학습 시작")
    model.fit(train_matrix)
    print("ALS 학습 완료")

    print(f"사용자 잠재벡터: {model.user_factors.shape}")
    print(f"게임 잠재벡터: {model.item_factors.shape}")

    return model


# ============================================================
# 4. Validation 사용자별 게임 목록
# ============================================================

def group_games_by_user(df):
    """
    사용자별 게임 ID 집합을 생성합니다.
    """

    return (
        df.groupby("user_id")["unified_game_id"]
        .agg(set)
        .to_dict()
    )


# ============================================================
# 5. 배치 사용자 행렬 생성
# ============================================================

def build_batch_matrix(
    batch_users,
    observed_by_user,
    game_to_idx,
    num_games
):
    """
    Validation 사용자들의 Observed 게임을
    배치 단위 CSR 행렬로 변환합니다.

    Validation 사용자는 학습에 등장하지 않은 신규 사용자이므로
    Fold-in에 사용할 행렬을 별도로 생성합니다.
    """

    rows = []
    cols = []

    for local_user_idx, user_id in enumerate(batch_users):

        observed_games = observed_by_user[user_id]

        for game_id in observed_games:

            game_idx = game_to_idx.get(game_id)

            # 학습 데이터에 존재하는 게임만 사용
            if game_idx is not None:
                rows.append(local_user_idx)
                cols.append(game_idx)

    confidence = np.full(
        len(rows),
        1 + CONFIDENCE_ALPHA,
        dtype=np.float32
    )

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
# 6. 배치 Fold-in 추천
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
    여러 Validation 사용자에게 Fold-in을 적용하여
    Top-K 게임을 추천합니다.

    이미 관측된 소유 게임은 추천에서 제외합니다.
    """

    # 배치 행렬 생성
    batch_matrix = build_batch_matrix(
        batch_users,
        observed_by_user,
        game_to_idx,
        len(game_ids)
    )

    # 배치 내부 사용자 인덱스
    local_user_ids = np.arange(len(batch_users))

    # Fold-in 및 Top-K 추천
    item_indices, scores = model.recommend(
        userid=local_user_ids,
        user_items=batch_matrix,
        N=k,
        filter_already_liked_items=True,
        recalculate_user=True
    )

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
# 7. 사용자별 평가
# ============================================================

def evaluate_user(recommendations, heldout_games, k=10):
    """
    사용자 한 명의 추천 결과를 평가합니다.

    평가 지표:
        Recall@10
        NDCG@10
        Hit@10
    """

    recommended_games = [
        game_id
        for game_id, _ in recommendations[:k]
    ]

    # 추천 결과의 정답 여부
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
# 8. 전체 Steam Validation 평가
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
    모든 Validation 사용자를 배치 단위로 Fold-in하여 평가합니다.

    사용자별 지표를 계산한 뒤 평균을 구할 수 있도록
    DataFrame으로 반환합니다.
    """

    observed_by_user = group_games_by_user(val_observed)
    heldout_by_user = group_games_by_user(val_heldout)

    # 평가 사용자 일치 여부 확인
    if set(observed_by_user) != set(heldout_by_user):
        raise ValueError(
            "Observed와 Heldout의 사용자 집합이 다릅니다."
        )

    user_ids = list(observed_by_user.keys())
    num_users = len(user_ids)

    results = []

    print("\nSteam Validation 전체 사용자 평가 시작")
    print(f"평가 사용자 수: {num_users:,}명")
    print(f"배치 크기: {batch_size}")

    # 배치 단위 평가
    for start in range(0, num_users, batch_size):

        batch_users = user_ids[
            start:start + batch_size
        ]

        # Fold-in을 통한 추천 생성
        batch_recommendations = recommend_batch(
            model,
            batch_users,
            observed_by_user,
            game_ids,
            game_to_idx,
            k=k
        )

        # 사용자별 지표 계산
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

            # Observed / Heldout 중복 검사
            if not observed_games.isdisjoint(heldout_games):
                raise ValueError(
                    f"사용자 {user_id}: Observed와 Heldout 중복"
                )

            # 이미 관측한 게임 추천 여부 검사
            if not set(recommended_games).isdisjoint(
                observed_games
            ):
                raise ValueError(
                    f"사용자 {user_id}: Observed 게임이 추천됨"
                )

            # 추천 중복 검사
            if len(recommended_games) != len(
                set(recommended_games)
            ):
                raise ValueError(
                    f"사용자 {user_id}: 추천 게임 중복"
                )

            # 평가
            metrics = evaluate_user(
                recommendations,
                heldout_games,
                k=k
            )

            results.append({
                "user_id": user_id,
                **metrics
            })

        # 진행 상황 표시
        completed = min(
            start + batch_size,
            num_users
        )

        print(
            f"\r평가 진행: {completed:,}/{num_users:,}명",
            end="",
            flush=True
        )

    print("\n전체 사용자 평가 완료")

    return pd.DataFrame(results)


# ============================================================
# 9. 메인 실행
# ============================================================

def main():

    # 1. 데이터 불러오기
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

    print(f"학습 데이터: {len(train_df):,}건")
    print(f"Validation Observed: {len(val_observed):,}건")
    print(f"Validation Heldout: {len(val_heldout):,}건")

    # 2. 학습 데이터 중복 검사
    if train_df.duplicated(
        ["user_id", "unified_game_id"]
    ).any():
        raise ValueError(
            "학습 데이터에 중복 사용자-게임 관계가 존재합니다."
        )

    # 3. 학습 행렬 생성
    train_matrix, game_ids, game_to_idx = (
        build_train_matrix(train_df)
    )

    print(f"\n학습 행렬 크기: {train_matrix.shape}")
    print(f"추천 후보 게임: {len(game_ids):,}개")

    # 4. ALS 학습
    model = train_als(train_matrix)

    # 5. 전체 Steam Validation 평가
    results_df = evaluate_all_users(
        model,
        val_observed,
        val_heldout,
        game_ids,
        game_to_idx,
        k=TOP_K,
        batch_size=BATCH_SIZE
    )

    # 6. 사용자별 평가 지표 평균
    mean_metrics = results_df[
        ["recall_at_10", "ndcg_at_10", "hit_at_10"]
    ].mean()

    # 7. 결과 출력
    print("\n" + "=" * 55)
    print("Binary ALS - Steam Validation 결과")
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
