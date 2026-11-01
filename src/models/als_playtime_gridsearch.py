
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

from als_playtime import (
    OWNER_ALPHA,
    build_validation_matrix,
    evaluate_all_users,
    validate_interactions,
    validate_train_playtime
)


# ============================================================
# 1. 설정
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULT_DIR = PROJECT_DIR / "data" / "results"

RESULT_PATH = RESULT_DIR / "als_playtime_gridsearch.csv"

TOP_K = 10
BATCH_SIZE = 256
RANDOM_STATE = 42

# ALS 하이퍼파라미터
FACTOR_GRID = [24]
REGULARIZATION_GRID = [5.0]
ITERATION_GRID = [40]

# 플레이타임 가중치
PLAYTIME_BETA_GRID = [
    0.1,
    0.25,
    0.5,
    1.0
]

# 플레이타임 상한값 분위수
PLAYTIME_CAP_PERCENTILE_GRID = [
    0.50,
    0.75,
    0.90,
    0.95,
    0.99
]


# ============================================================
# 2. 학습 데이터 준비
# ============================================================

def prepare_train_data(train_df):
    """
    Steam Train의 사용자·게임 인덱스를 생성합니다.

    플레이타임 원본은 시간 단위로 변환하여 보관합니다.
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

    # 분 → 시간
    playtime_hours = (
        train_df["playtime_forever"]
        .to_numpy(dtype=np.float64) / 60
    )

    # 게임 ID → 행렬 인덱스
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
# 3. 상한값 후보 계산
# ============================================================

def calculate_cap_grid(playtime_hours):
    """
    Steam Train의 양수 플레이타임을 기준으로
    분위수별 상한값을 계산합니다.
    """

    positive_hours = playtime_hours[
        playtime_hours > 0
    ]

    if len(positive_hours) == 0:
        raise ValueError(
            "학습 데이터에 양수 플레이타임이 없습니다."
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
# 4. Weighted ALS 학습 행렬 생성
# ============================================================

def build_train_matrix(
    user_indices,
    game_indices,
    log_playtime,
    matrix_shape,
    beta
):
    """
    지정된 Beta와 로그 플레이타임으로
    학습용 CSR 행렬을 생성합니다.

    c_ui = 1 + OWNER_ALPHA + beta * log_playtime
    """

    confidence = (
        1
        + OWNER_ALPHA
        + beta * log_playtime
    ).astype(np.float32)

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
# 5. 기존 실험 결과 불러오기
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
    중복 실험 확인을 위한 파라미터 키입니다.

    실수 비교 오차를 줄이기 위해 반올림합니다.
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
    기존 CSV를 불러오고 완료된 실험 조합을 추출합니다.
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
            f"기존 CSV에 필수 컬럼이 없습니다: {missing}"
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
    Beta와 플레이타임 상한값을 함께 탐색합니다.

    기존에 완료한 실험은 건너뛰고
    새로운 결과만 CSV에 추가합니다.
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

    # 기존 실험 결과 확인
    existing_df, completed_params = load_existing_results()

    # --------------------------------------------------------
    # 1. 전체 실험 조합 생성
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
    # 2. 이미 완료한 조합 제외
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

    print(f"기존 실험 결과: {len(existing_df)}개")
    print(f"전체 탐색 조합: {len(combinations)}개")
    print(f"새로 실행할 조합: {len(remaining)}개")
    print(f"평가 사용자 수: {len(val_user_ids):,}명")

    total_start = perf_counter()

    # 현재 계산된 상한값
    current_cap = None
    log_playtime = None

    # --------------------------------------------------------
    # 3. 실험 수행
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
            f"실험 {experiment_idx}/{len(remaining)}"
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
        # 4. 상한값별 로그 플레이타임 계산
        # ----------------------------------------------------

        # 상한값이 바뀌었을 때만 다시 계산
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
        # 5. Beta별 학습 행렬 생성
        # ----------------------------------------------------

        train_matrix = build_train_matrix(
            user_indices,
            game_indices,
            log_playtime,
            matrix_shape,
            beta
        )

        # ----------------------------------------------------
        # 6. ALS 모델 생성 및 학습
        # ----------------------------------------------------

        model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            alpha=1.0,
            random_state=RANDOM_STATE,
            dtype=np.float32
        )

        model.fit(
            train_matrix,
            show_progress=False
        )

        # ----------------------------------------------------
        # 7. Steam Validation 평가
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
        # 8. 결과 기록
        # ----------------------------------------------------

        # 기존 CSV와 동일한 컬럼 구조 유지
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

        print("\n실험 결과")

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

        print(f"실행 시간: {elapsed_seconds:.2f}초")

        # ----------------------------------------------------
        # 9. 기존 CSV에 결과 추가
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

        # 완료 조합 등록
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

        print("CSV 저장 완료")

    # --------------------------------------------------------
    # 10. 누적 결과 불러오기
    # --------------------------------------------------------

    total_elapsed = perf_counter() - total_start

    all_results = pd.read_csv(
        RESULT_PATH,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("Grid Search 완료")
    print("=" * 60)

    print(f"이번 실행 시간: {total_elapsed:.2f}초")
    print(f"누적 실험 결과: {len(all_results)}개")

    return all_results


# ============================================================
# 7. 최적 조합 출력
# ============================================================

def print_best_result(results_df):
    """
    NDCG@10 기준으로 최적 결과를 출력합니다.

    Beta=0은 Binary ALS와 동일하므로
    Beta>0인 Weighted ALS 결과도 별도로 확인합니다.
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
    print("전체 Grid Search 최종 결과")
    print("=" * 60)

    print("\nNDCG@10 기준 상위 10개 조합")

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

    # 백분율로 변환
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
    # Beta>0인 Weighted ALS 최적 결과
    # --------------------------------------------------------

    weighted_results = sorted_results[
        sorted_results["playtime_beta"] > 0
    ]

    if weighted_results.empty:
        print("\nBeta>0인 실험 결과가 없습니다.")
        return

    best = weighted_results.iloc[0]

    print("\n" + "=" * 60)
    print("Playtime Weighted ALS 최적 하이퍼파라미터")
    print("=" * 60)

    print(f"Factors: {int(best['factors'])}")
    print(f"Regularization: {best['regularization']}")
    print(f"Iterations: {int(best['iterations'])}")
    print(f"Owner Alpha: {int(best['owner_alpha'])}")
    print(f"Playtime Beta: {best['playtime_beta']}")
    print(f"Playtime Cap: {best['cap_hours']:.2f}시간")

    print("\n최적 Weighted ALS 평가 성능")

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
    # Binary ALS와 비교
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

        print("\nBinary ALS와 비교")

        print(
            f"Binary NDCG@10: "
            f"{best_binary['ndcg_at_10'] * 100:.2f}%"
        )

        print(
            f"Weighted NDCG@10: "
            f"{best['ndcg_at_10'] * 100:.2f}%"
        )

        print(
            f"NDCG 차이: "
            f"{ndcg_difference:+.2f}%p"
        )

    print(f"\n전체 결과 저장: {RESULT_PATH}")


# ============================================================
# 8. 메인 실행
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

    train_df = validate_train_playtime(
        train_df
    )

    print("=" * 60)
    print("Playtime Weighted ALS - Grid Search")
    print("=" * 60)

    print(f"학습 데이터: {len(train_df):,}건")
    print(f"Validation Observed: {len(val_observed):,}건")
    print(f"Validation Heldout: {len(val_heldout):,}건")

    # --------------------------------------------------------
    # 3. 학습 데이터 준비
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

    print(f"\n학습 행렬 크기: {matrix_shape}")
    print(f"추천 후보 게임: {len(game_ids):,}개")

    # --------------------------------------------------------
    # 4. 상한값 후보 계산
    # --------------------------------------------------------

    cap_grid = calculate_cap_grid(
        playtime_hours
    )

    print("\n플레이타임 상한값 후보")

    for percentile, cap_hours in cap_grid:

        print(
            f"{percentile * 100:.0f}% 분위수: "
            f"{cap_hours:.2f}시간"
        )

    # --------------------------------------------------------
    # 5. Validation Fold-in 행렬 생성
    # --------------------------------------------------------

    # 평가에서는 소유 정보만 사용합니다.
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
    # 7. 최적 결과 출력
    # --------------------------------------------------------

    print_best_result(
        results_df
    )


if __name__ == "__main__":
    main()
