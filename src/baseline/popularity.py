"""
Popularity baseline: rank Steam Train games by distinct owning users.
"""

from pathlib import Path

import pandas as pd

from src.evaluation.metrics import ranking_metrics

K = 10 # top K games to recommend
DATA = Path("data/processed") # data directory
RESULTS = Path("results/popularity") # results directory


def build_popularity(train: pd.DataFrame) -> pd.DataFrame:
    required = {"user_id", "unified_game_id"}
    if not required.issubset(train.columns):
        raise ValueError(f"Steam train is missing: {required - set(train.columns)}")

    # count the number of unique users for each game
    counts = (
        train.groupby("unified_game_id")["user_id"]
        .nunique()
        .reset_index(name="ownership_count")
    )
    # sort the games by the number of unique users in descending order
    return counts.sort_values(
        ["ownership_count", "unified_game_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


def evaluate_popularity(
    ranking: list[str], observed: pd.DataFrame, heldout: pd.DataFrame, split: str
) -> dict:
    observed_by_user = observed.groupby("user_id")["unified_game_id"].agg(set).to_dict()
    heldout_by_user = heldout.groupby("user_id")["unified_game_id"].agg(set).to_dict()

    if set(observed_by_user) != set(heldout_by_user):
        raise ValueError(f"{split}: observed/held-out user sets differ")

    totals = {f"recall@{K}": 0.0, f"ndcg@{K}": 0.0, f"hit_rate@{K}": 0.0}
    for user, hidden in heldout_by_user.items():
        seen = observed_by_user[user]
        if seen & hidden:
            raise ValueError(f"{split}: user {user} has leaked held-out games")

        # The global ranking is shared; remove already-owned games per user.
        recommended = []
        for game in ranking:
            if game not in seen:
                # add the game to the recommended list if it is not in the observed games.
                recommended.append(game)
                if len(recommended) == K:
                    break

        scores = ranking_metrics(recommended, hidden, K) # evaluate the popularity baseline
        for metric, value in scores.items():
            totals[metric] += value

    n = len(heldout_by_user)
    if n == 0:
        raise ValueError(f"{split}: no evaluation users")
    return {"split": split, "users": n, **{m: total / n for m, total in totals.items()}} # return the average metrics


def main() -> None:
    steam_train = pd.read_parquet(DATA / "steam_train.parquet")
    popularity = build_popularity(steam_train)
    ranking = popularity["unified_game_id"].tolist()

    rows = []
    for split, prefix in (
        ("steam_validation", "steam_val"),
        ("playstation_test", "ps_test"),
    ):
        observed = pd.read_parquet(DATA / f"{prefix}_observed.parquet")
        heldout = pd.read_parquet(DATA / f"{prefix}_heldout.parquet")
        rows.append(evaluate_popularity(ranking, observed, heldout, split)) # evaluate the popularity baseline

    # save the results
    RESULTS.mkdir(parents=True, exist_ok=True)
    popularity.to_csv(RESULTS / "steam_game_popularity.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "popularity_metrics.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved: results/steam_game_popularity.csv")
    print("Saved: results/popularity_metrics.csv")


if __name__ == "__main__":
    main()
