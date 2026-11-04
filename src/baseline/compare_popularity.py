"""
Compare Steam Train and PlayStation Test ownership popularity.
"""

import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

PROCESSED = Path('data/processed')
INTERIM = Path('data/interim')
RESULTS = Path('results')


def read_interactions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}; run split_data.py first.')
    df = pd.read_parquet(path, columns=['user_id', 'unified_game_id'])
    if df[['user_id', 'unified_game_id']].isna().any().any():
        raise ValueError(f'Null user/game IDs in {path}')
    return df.drop_duplicates(['user_id', 'unified_game_id'])


def main() -> None:
    # read the interactions
    steam = read_interactions(PROCESSED / 'steam_train.parquet')
    # read the PlayStation(test set) observe and held out interactions
    ps = pd.concat(
        [read_interactions(PROCESSED / 'ps_test_observed.parquet'),
         read_interactions(PROCESSED / 'ps_test_heldout.parquet')],
        ignore_index=True,
    ).drop_duplicates(['user_id', 'unified_game_id'])

    # Evaluate ranking differences on exactly the same recommendable game catalog.
    catalog = sorted(steam['unified_game_id'].unique())
    ps = ps.loc[ps['unified_game_id'].isin(catalog)].copy()

    # count the number of users for each game
    steam_n = steam['user_id'].nunique()
    ps_n = ps['user_id'].nunique()
    if not steam_n or not ps_n:
        raise ValueError('No users to compare.')

    # Game | User count Frequency Table
    steam_counts = steam.groupby('unified_game_id')['user_id'].nunique()
    ps_counts = ps.groupby('unified_game_id')['user_id'].nunique()

    # create a dataframe to store the results
    result = pd.DataFrame({'unified_game_id': catalog})
    result['steam_users'] = result['unified_game_id'].map(steam_counts).fillna(0).astype(int)
    result['ps_users'] = result['unified_game_id'].map(ps_counts).fillna(0).astype(int)
    result['steam_share'] = result['steam_users'] / steam_n
    result['ps_share'] = result['ps_users'] / ps_n

    # Deterministic tie-breaking by the game ID.
    steam_order = result.sort_values(['steam_users', 'unified_game_id'], ascending=[False, True])
    ps_order = result.sort_values(['ps_users', 'unified_game_id'], ascending=[False, True]) 

    steam_rank = dict(zip(steam_order['unified_game_id'], range(1, len(result) + 1)))
    ps_rank = dict(zip(ps_order['unified_game_id'], range(1, len(result) + 1)))

    result['steam_rank'] = result['unified_game_id'].map(steam_rank)
    result['ps_rank'] = result['unified_game_id'].map(ps_rank)
    result['rank_shift'] = result['steam_rank'] - result['ps_rank']  # positive: more popular on PS

    shared_path = INTERIM / 'shared_games.parquet'
    if shared_path.exists():
        # game title mapping
        meta = pd.read_parquet(shared_path, columns=['unified_game_id', 'steam_title'])
        meta = meta.drop_duplicates('unified_game_id')
        result = result.merge(meta, on='unified_game_id', how='left', validate='one_to_one')
    else:
        result['steam_title'] = result['unified_game_id']

    # calculate the top-k overlap and Jaccard similarity
    # Overlap@K - the proportion of games in the top K that are shared between the two lists
    # Jaccard - the similarity of the two popular game sets
    tops = {}
    for k in (10, 20, 50, 100):
        a = set(steam_order['unified_game_id'].head(k))
        b = set(ps_order['unified_game_id'].head(k))

        intersection = len(a & b)
        tops[str(k)] = {
            'shared_games': intersection,
            'overlap_at_k': intersection / min(k, len(catalog)),
            'jaccard': intersection / len(a | b),
        }

    # Calculate Spearman correlation for all games
    rho, pvalue = spearmanr(
        result['steam_users'],
        result['ps_users']
    )

    # Calculate Spearman correlation for Steam Top-K games
    top_k_spearman = {}
    for k in (20, 50, 100):
        top_k = result.nsmallest(k, 'steam_rank')

        top_rho, top_pvalue = spearmanr(
            top_k['steam_users'],
            top_k['ps_users']
        )

        top_k_spearman[str(k)] = {
            'spearman_rank_correlation': float(top_rho),
            'spearman_p_value': float(top_pvalue),
        }

    summary = {
        'steam_train_users': int(steam_n),
        'ps_test_users': int(ps_n),
        'compared_games': len(catalog),
        'spearman_rank_correlation': float(rho),
        'spearman_p_value': float(pvalue),
        'top_k_overlap': tops,
        'steam_top_k_spearman': top_k_spearman,
        'important_note': (
            'PS popularity uses observed + held-out only for POST-HOC diagnosis. '
            'Do not use PS ranking to train, tune or rerank a recommender.'
        ),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    result = result.sort_values('steam_rank').reset_index(drop=True)
    result.to_csv(RESULTS / 'popularity_comparison.csv', index=False, encoding='utf-8-sig')
    summary_path = RESULTS / 'popularity_comparison_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    top20 = pd.concat([
        result.nsmallest(20, 'steam_rank').assign(list_source='steam_top20'),
        result.nsmallest(20, 'ps_rank').assign(list_source='ps_top20'),
    ], ignore_index=True)
    top20.to_csv(RESULTS / 'top20_popularity_comparison.csv', index=False, encoding='utf-8-sig')

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('\nSteam Top 20 (with PS rankings):')
    print(result.nsmallest(20, 'steam_rank')[['steam_rank', 'ps_rank', 'steam_title', 'steam_share', 'ps_share']].to_string(index=False))
    print('\nPS Top 20 (with Steam rankings):')
    print(result.nsmallest(20, 'ps_rank')[['ps_rank', 'steam_rank', 'steam_title', 'ps_share', 'steam_share']].to_string(index=False))
    print('\nSaved 3 files in results/.')


if __name__ == '__main__':
    main()
