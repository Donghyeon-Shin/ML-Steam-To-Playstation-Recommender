
import os

# OpenBLAS 스레드 설정
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
# 1. 설정
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULT_DIR = PROJECT_DIR / "data" / "results"

RESULT_PATH = RESULT_DIR / "als_binary_gridsearch.csv"

TOP_K = 10
BATCH_SIZE = 256
ITERATIONS = 40
RANDOM_STATE = 42

# 2차 Grid Search 탐색 범위
FACTOR_GRID = [24]
REGULARIZATION_GRID = [5.0]
ALPHA_GRID = [2]


# ============================================================
# 2. Validation 평가
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
    전체 Steam Validation 사용자를 배치 단위로 평가합니다.

    신규 사용자의 Observed 게임으로 Fold-in을 수행하고,
    Top-10 추천 결과를 평가합니다.
    """

    user_ids = list(observed_by_user.keys())
    num_users = len(user_ids)
    num_games = len(game_ids)

    metric_sums = {
        "recall_at_10": 0.0,
        "ndcg_at_10": 0.0,
        "hit_at_10": 0.0
    }

    # 배치 단위 처리
    for start in range(0, num_users, BATCH_SIZE):

        batch_users = user_ids[start:start + BATCH_SIZE]

        rows = []
        cols = []

        # 배치 사용자 × 게임 행렬 생성
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

        batch_matrix = csr_matrix(
            (
                confidence,
                (rows, cols)
            ),
            shape=(len(batch_users), num_games),
            dtype=np.float32
        )

        # Fold-in Top-10 추천
        item_indices, scores = model.recommend(
            userid=np.arange(len(batch_users)),
            user_items=batch_matrix,
            N=TOP_K,
            filter_already_liked_items=True,
            recalculate_user=True
        )

        # 사용자별 평가
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

            # Observed / Heldout 중복 검사
            if not observed_games.isdisjoint(heldout_games):
                raise ValueError(
                    f"사용자 {user_id}: Observed/Heldout 중복"
                )

            # Observed 게임 추천 여부 검사
            if not recommended_games.isdisjoint(observed_games):
                raise ValueError(
                    f"사용자 {user_id}: Observed 게임 추천"
                )

            # 추천 중복 검사
            if len(recommended_games) != len(recommendations):
                raise ValueError(
                    f"사용자 {user_id}: 추천 게임 중복"
                )

            # 평가
            metrics = evaluate_user(
                recommendations,
                heldout_games,
                k=TOP_K
            )

            for metric_name in metric_sums:
                metric_sums[metric_name] += metrics[metric_name]

    # 사용자별 평가 지표의 평균
    return {
        metric_name: total / num_users
        for metric_name, total in metric_sums.items()
    }


# ============================================================
# 3. 기존 실험 결과 불러오기
# ============================================================

def load_existing_results():
    """
    기존 CSV 결과를 불러오고
    이미 완료한 하이퍼파라미터 조합을 반환합니다.
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
    Grid Search를 수행합니다.

    기존 CSV에 존재하는 조합은 건너뛰고,
    새로운 실험 결과만 CSV에 추가합니다.
    """

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 기존 결과 확인
    existing_df, completed_params = load_existing_results()

    # 전체 하이퍼파라미터 조합
    combinations = list(product(
        FACTOR_GRID,
        REGULARIZATION_GRID,
        ALPHA_GRID
    ))

    total_combinations = len(combinations)

    # 아직 실행하지 않은 조합
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

    print(f"기존 실험 결과: {len(existing_df)}개")
    print(f"현재 탐색 조합: {total_combinations}개")
    print(f"새로 실행할 조합: {len(remaining_combinations)}개")
    print(f"평가 사용자: {len(observed_by_user):,}명")
    print(f"추천 개수: Top-{TOP_K}")
    print(f"Iterations: {ITERATIONS}")

    total_start_time = perf_counter()

    # 새 실험만 수행
    for experiment_idx, (
        factors,
        regularization,
        confidence_alpha
    ) in enumerate(remaining_combinations, start=1):

        print("\n" + "-" * 60)
        print(
            f"새 실험 {experiment_idx}/"
            f"{len(remaining_combinations)}"
        )
        print("-" * 60)

        print(f"Factors: {factors}")
        print(f"Regularization: {regularization}")
        print(f"Confidence Alpha: {confidence_alpha}")

        start_time = perf_counter()

        # ----------------------------------------------------
        # 1. Confidence 업데이트
        # ----------------------------------------------------

        train_matrix.data.fill(
            1 + confidence_alpha
        )

        # ----------------------------------------------------
        # 2. 모델 생성 및 학습
        # ----------------------------------------------------

        model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=ITERATIONS,
            alpha=1.0,
            random_state=RANDOM_STATE,
            dtype=np.float32
        )

        model.fit(
            train_matrix,
            show_progress=False
        )

        # ----------------------------------------------------
        # 3. 전체 Validation 평가
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
        # 4. 결과 기록
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

        print(f"실행 시간: {elapsed_seconds:.2f}초")

        # ----------------------------------------------------
        # 5. CSV에 새로운 결과만 추가
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

        # 완료한 조합 기록
        completed_params.add((
            factors,
            regularization,
            confidence_alpha,
            ITERATIONS
        ))

        print("CSV 저장 완료")

    total_elapsed = perf_counter() - total_start_time

    print("\n" + "=" * 60)
    print("Grid Search 실행 완료")
    print("=" * 60)

    print(f"이번 실행 시간: {total_elapsed:.2f}초")

    # 기존 + 신규 결과 모두 불러오기
    all_results = pd.read_csv(
        RESULT_PATH,
        encoding="utf-8-sig"
    )

    print(f"누적 실험 결과: {len(all_results)}개")

    return all_results


# ============================================================
# 5. 최적 조합 출력
# ============================================================

def print_best_result(results_df):
    """
    누적된 전체 실험 결과를 NDCG@10 기준으로 정렬하고
    최적 하이퍼파라미터를 출력합니다.
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
    print("전체 Grid Search 최종 결과")
    print("=" * 60)

    print("\nNDCG@10 기준 상위 10개 조합")

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

    # 백분율 변환
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
    print("최적 하이퍼파라미터")
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

    print("\n최적 모델 평가 성능")

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

    print(f"\n전체 결과 저장: {RESULT_PATH}")


# ============================================================
# 6. 메인 실행
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

    print("Steam 데이터 로드 완료")
    print(f"학습 데이터: {len(train_df):,}건")

    # 2. 학습 행렬 생성
    train_matrix, game_ids, game_to_idx = (
        build_train_matrix(train_df)
    )

    print(f"학습 행렬: {train_matrix.shape}")

    # 3. Validation 사용자별 게임 목록 생성
    observed_by_user = group_games_by_user(val_observed)
    heldout_by_user = group_games_by_user(val_heldout)

    # 사용자 집합 검사
    if set(observed_by_user) != set(heldout_by_user):
        raise ValueError(
            "Observed와 Heldout의 사용자 집합이 다릅니다."
        )

    # 4. Grid Search 수행
    results_df = grid_search(
        train_matrix,
        observed_by_user,
        heldout_by_user,
        game_ids,
        game_to_idx
    )

    # 5. 누적 결과에서 최적 모델 선정
    print_best_result(results_df)


if __name__ == "__main__":
    main()
