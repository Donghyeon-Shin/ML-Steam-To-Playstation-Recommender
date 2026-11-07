"""Evaluate the Steam-validation winner (trial 34) on PlayStation."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .evaluate import evaluate, read_frame
from .tuned_model import TunedItemBasedCF


# Frozen from tuning_10/best_config.json, overall.
PARAMETERS = {
    "weighting": "binary",
    "neighbors": 20,
    "shrinkage": 100,
    "min_common_users": 20,
    "min_similarity": 0,
}
K = 10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--output-dir", type=Path, default=Path("results/item_based/final_ps_10"))
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Choose a new or empty output directory")

    train = read_frame(args.data_dir, "steam_train", args.format)
    observed = read_frame(args.data_dir, "ps_test_observed", args.format)
    heldout = read_frame(args.data_dir, "ps_test_heldout", args.format)

    model = TunedItemBasedCF(**PARAMETERS).fit(train)
    metrics, per_user = evaluate(model, observed, heldout, k=K)
    result = {
        "split": "playstation_test",
        "validation_trial": 34,
        "candidate_games": len(model.items),
        "k": K,
        **PARAMETERS,
        **metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(args.output_dir / "metrics.csv", index=False)
    per_user.to_csv(args.output_dir / "per_user.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
