"""Staged search on Steam validation only. Never reads PlayStation files."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter
import pandas as pd
from src.models.item_based.evaluate import read_frame, evaluate
from src.models.item_based.tuned_model import TunedItemBasedCF


def choose_best(records):
    # Exact ties preserve the predeclared enumeration order.
    return max(records, key=lambda row: row['ndcg@10'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data/processed'))
    p.add_argument('--format', choices=['csv', 'parquet'], default='parquet')
    p.add_argument('--output-dir', type=Path, default=Path('results/item_based/tuning_v1'))
    p.add_argument('--neighbors', type=int, nargs='+', default=[0, 20, 50, 100, 200, 500])
    p.add_argument('--shrinkage', type=float, nargs='+', default=[0, 10, 50, 100])
    p.add_argument('--betas', type=float, nargs='+', default=[0, .1, .25, .5, 1])
    p.add_argument('--cap-quantiles', type=float, nargs='+', default=[.5, .75, .9, .99])
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--min-common-users', type=int, nargs='+', default=[0, 5, 20, 50])
    p.add_argument('--min-similarity', type=float, nargs='+', default=[0, .01, .05, .1])
    args = p.parse_args()
    # Require a fresh destination so a rerun cannot overwrite an earlier experiment.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error('Use an empty output directory to preserve previous experiments')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Use a fixed Steam split for every trial: fit on train, score from observed
    # ownership, and reserve held-out games for the validation metrics.
    frames = [read_frame(args.data_dir, name, args.format) for name in
              ['steam_train', 'steam_val_observed', 'steam_val_heldout']]
    train, observed, heldout = frames
    # Record input fingerprints so later runs can verify that the split files match.
    hashes = {}
    for name in ['steam_train', 'steam_val_observed', 'steam_val_heldout']:
        path = args.data_dir / f'{name}.{args.format}'
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {'selection_metric': 'ndcg@10', 'tie_rule': 'first enumerated',
                'search': 'staged, not exhaustive joint search', 'data_sha256': hashes,
                'options': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (args.output_dir / 'search_config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    records = []

    # Evaluate one setting and record ranking metrics, graph size, and runtime.
    def run(stage, **config):
        start = perf_counter()
        model = TunedItemBasedCF(**config).fit(train)
        fit_seconds = perf_counter() - start
        metrics, users = evaluate(model, observed, heldout, k=10, batch_size=args.batch_size)
        row = {'trial': len(records), 'stage': stage, **config,
               'cap_value': model.cap_value_, 'candidate_games': len(model.items), **metrics,
               'raw_edges': model.raw_edges_, 'retained_edges': model.retained_edges_,
               'retained_edge_ratio': model.retained_edges_ / model.raw_edges_ if model.raw_edges_ else 0,
               'zero_neighbor_games': model.zero_neighbor_games_,
               'fit_seconds': fit_seconds, 'total_seconds': perf_counter() - start}
        records.append(row)
        # Persist completed trials as the search runs, along with per-user metrics.
        pd.DataFrame(records).to_csv(args.output_dir / 'grid_results.csv', index=False)
        users.to_csv(args.output_dir / f'trial_{row["trial"]}_per_user.csv', index=False)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return row

    # Reproduce both previous defaults separately from the new hours transform.
    run('reference', weighting='binary', neighbors=0, shrinkage=0)
    run('reference', weighting='playtime', neighbors=0, shrinkage=0,
        beta=1, cap_quantile=None, time_unit='minutes')
    # Stage 1: vary neighbor count and shrinkage on binary ownership.
    binary = []
    for n, shrink in itertools.product(args.neighbors, args.shrinkage):
        binary.append(run('binary_grid', weighting='binary', neighbors=n, shrinkage=shrink))
    best_binary = choose_best(binary)
    # Stage 2: hold the best binary neighborhood fixed while tuning pair filters.
    filtered_binary = []
    for common, threshold in itertools.product(args.min_common_users, args.min_similarity):
        filtered_binary.append(run('binary_filters', weighting='binary',
            neighbors=best_binary['neighbors'], shrinkage=best_binary['shrinkage'],
            min_common_users=common, min_similarity=threshold))
    best_filters = choose_best(filtered_binary)
    playtime = []
    # Fix neighborhood settings selected by binary search, then tune time weights.
    for beta in args.betas:
        # With beta zero, playtime cannot change the weights, so caps are redundant.
        caps = [None] if beta == 0 else [None, *args.cap_quantiles]
        for cap in caps:
            playtime.append(run('playtime_grid', weighting='playtime',
                                neighbors=best_binary['neighbors'], shrinkage=best_binary['shrinkage'],
                                min_common_users=best_filters['min_common_users'],
                                min_similarity=best_filters['min_similarity'],
                                beta=beta, cap_quantile=cap, time_unit='hours'))
    selected_time = choose_best(playtime)
    # Revisit neighbor count and shrinkage using the selected playtime settings.
    refined = []
    for n, shrink in itertools.product(args.neighbors, args.shrinkage):
        refined.append(run('playtime_refine', weighting='playtime', neighbors=n,
                           shrinkage=shrink, beta=selected_time['beta'],
                           min_common_users=best_filters['min_common_users'],
                           min_similarity=best_filters['min_similarity'],
                           cap_quantile=selected_time['cap_quantile'], time_unit='hours'))
    selected_playtime = choose_best(playtime + refined)
    # Tune the pair filters again for the best playtime neighborhood.
    filtered_playtime = []
    for common, threshold in itertools.product(args.min_common_users, args.min_similarity):
        filtered_playtime.append(run('playtime_filters', weighting='playtime',
            neighbors=selected_playtime['neighbors'], shrinkage=selected_playtime['shrinkage'],
            beta=selected_playtime['beta'], cap_quantile=selected_playtime['cap_quantile'],
            time_unit='hours', min_common_users=common, min_similarity=threshold))
    # Record the best setting within each weighting mode and across all trials.
    best = {'binary': choose_best(binary + filtered_binary),
            'playtime': choose_best(playtime + refined + filtered_playtime),
            'overall': choose_best(records)}
    (args.output_dir / 'best_config.json').write_text(json.dumps(best, indent=2), encoding='utf-8')
    from src.models.item_based.compare_results import create_report
    create_report(args.output_dir)
    print('Saved grid_results.csv, best_config.json and comparison.html', flush=True)


if __name__ == '__main__':
    main()
