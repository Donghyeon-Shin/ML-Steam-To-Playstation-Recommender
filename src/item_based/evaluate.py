"""Run one fixed Item-Based configuration on validation OR external test."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.evaluation.metrics import ranking_metrics
from src.item_based.model import ItemBasedCF


def read_frame(directory, stem, file_format):
    path = directory / f"{stem}.{file_format}"
    if file_format == "csv":
        return pd.read_csv(path, dtype={"user_id": "string", "unified_game_id": "string"})
    return pd.read_parquet(path)


def evaluate(model, observed, heldout, k=10, batch_size=256):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    frames = []
    for frame in (observed, heldout):
        frame = frame[["user_id", "unified_game_id"]].copy()
        if frame.isna().any().any():
            raise ValueError("Null evaluation IDs")
        frame = frame.astype(str)
        if set(frame["unified_game_id"]) - model.item_to_index.keys():
            raise ValueError("Evaluation contains games outside training catalog")
        frames.append(frame.groupby("user_id")["unified_game_id"].agg(set).to_dict())
    seen_by_user, hidden_by_user = frames
    if not hidden_by_user or seen_by_user.keys() != hidden_by_user.keys():
        raise ValueError("Empty evaluation or observed/held-out user sets differ")
    users = sorted(hidden_by_user)
    for user in users:
        if seen_by_user[user] & hidden_by_user[user]:
            raise ValueError(f"Observed/held-out overlap for user {user}")

    records = []
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        recs = model.recommend_many([seen_by_user[u] for u in batch], k=k)
        for user, recommended in zip(batch, recs):
            records.append({"user_id": user, **ranking_metrics(recommended, hidden_by_user[user], k)})
    per_user = pd.DataFrame(records)
    metrics = per_user.drop(columns="user_id").mean().to_dict()
    return {"users": len(users), **metrics}, per_user


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weighting", choices=["binary", "playtime"], required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--neighbors", type=int, default=0, help="0 = all item similarities")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--output-dir", type=Path, default=Path("results/item_based"))
    args = parser.parse_args()
    if args.k <= 0 or args.batch_size <= 0:
        parser.error("k and batch-size must be positive")

    train = read_frame(args.data_dir, "steam_train", args.format)
    prefix = "steam_val" if args.split == "validation" else "ps_test"
    observed = read_frame(args.data_dir, f"{prefix}_observed", args.format)
    heldout = read_frame(args.data_dir, f"{prefix}_heldout", args.format)
    print(f"Training {args.weighting}: {len(train):,} interactions", flush=True)
    start = perf_counter()
    model = ItemBasedCF(args.weighting, args.neighbors).fit(train)
    fit_seconds = perf_counter() - start
    del train
    print(f"Evaluating {args.split}: {len(model.items):,} candidates", flush=True)
    start = perf_counter()
    summary, per_user = evaluate(model, observed, heldout, args.k, args.batch_size)
    summary.update({
        "model": f"item_based_{args.weighting}", "split": args.split,
        "neighbors": args.neighbors, "k": args.k, "candidate_games": len(model.items),
        "fit_seconds": fit_seconds, "evaluation_seconds": perf_counter() - start,
        "playtime_formula": "1 + ln(1 + minutes) on owned pairs" if args.weighting == "playtime" else None,
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.weighting}_{args.split}_neighbors{args.neighbors}_k{args.k}"
    pd.DataFrame([summary]).to_csv(args.output_dir / f"{name}_metrics.csv", index=False)
    per_user.to_csv(args.output_dir / f"{name}_per_user.csv", index=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output_dir / f"{name}_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output_dir / name}_metrics.csv")


if __name__ == "__main__":
    main()
