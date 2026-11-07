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
    # The baseline only needs ownership IDs; each owner casts one vote per game.
    required = {"user_id", "unified_game_id"}
    if not required.issubset(train.columns):
        raise ValueError(f"Steam train is missing: {required - set(train.columns)}")

    # Count distinct owners so duplicate rows cannot increase a game's rank.
    counts = (
        train.groupby("unified_game_id")["user_id"]
        .nunique()
        .reset_index(name="ownership_count")
    )
    # Break ties by unified game ID so both evaluation splits use a reproducible ranking.
    return counts.sort_values(
        ["ownership_count", "unified_game_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


def evaluate_popularity(
    ranking: list[str], observed: pd.DataFrame, heldout: pd.DataFrame, split: str
) -> dict:
    # Keep observed libraries separate from the hidden games used as ground truth.
    observed_by_user = observed.groupby("user_id")["unified_game_id"].agg(set).to_dict()
    heldout_by_user = heldout.groupby("user_id")["unified_game_id"].agg(set).to_dict()

    # A missing profile or answer set would make the corresponding user unevaluable.
    if set(observed_by_user) != set(heldout_by_user):
        raise ValueError(f"{split}: observed/held-out user sets differ")

    totals = {f"recall@{K}": 0.0, f"ndcg@{K}": 0.0, f"hit_rate@{K}": 0.0}
    for user, hidden in heldout_by_user.items():
        seen = observed_by_user[user]
        # Reject overlap so known games never appear in the same user's ground truth.
        if seen & hidden:
            raise ValueError(f"{split}: user {user} has leaked held-out games")

        # The global ranking is shared; remove already-owned games per user.
        recommended = []
        for game in ranking:
            if game not in seen:
                recommended.append(game)
                if len(recommended) == K:
                    break

        scores = ranking_metrics(recommended, hidden, K)
        # Accumulate one score per user so users with large libraries carry equal weight.
        for metric, value in scores.items():
            totals[metric] += value

    n = len(heldout_by_user)
    if n == 0:
        raise ValueError(f"{split}: no evaluation users")
    return {"split": split, "users": n, **{m: total / n for m, total in totals.items()}}


def main() -> None:
    steam_train = pd.read_parquet(DATA / "steam_train.parquet")
    popularity = build_popularity(steam_train)
    ranking = popularity["unified_game_id"].tolist()

    rows = []
    # Reuse the Steam-derived ranking for both splits; only the user histories change.
    for split, prefix in (
        ("steam_validation", "steam_val"),
        ("playstation_test", "ps_test"),
    ):
        observed = pd.read_parquet(DATA / f"{prefix}_observed.parquet")
        heldout = pd.read_parquet(DATA / f"{prefix}_heldout.parquet")
        rows.append(evaluate_popularity(ranking, observed, heldout, split))

    RESULTS.mkdir(parents=True, exist_ok=True)
    # Save the ranking for inspection and one metric row per evaluation split.
    popularity.to_csv(RESULTS / "steam_game_popularity.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "popularity_metrics.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved: results/steam_game_popularity.csv")
    print("Saved: results/popularity_metrics.csv")


if __name__ == "__main__":
    main()
