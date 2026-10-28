"""
Step 04: create leak-free Steam training/validation and PlayStation test splits.

Inputs:  data/interim/steam_clean.parquet, data/interim/ps_clean.parquet
Outputs: data/processed/steam_train.parquet, data/processed/steam_val_observed.parquet,
         data/processed/steam_val_heldout.parquet, data/processed/ps_test_observed.parquet,
         data/processed/ps_test_heldout.parquet, data/processed/split_report.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# Check if the interactions are valid
def check_interactions(df: pd.DataFrame, platform: str) -> None:
    required = {'user_id', 'unified_game_id'}
    if not required.issubset(df.columns):
        raise ValueError(f'{platform}: missing columns {required - set(df.columns)}')
    if df[list(required)].isna().any().any():
        raise ValueError(f'{platform}: null user/game IDs found')
    if df.duplicated(['user_id', 'unified_game_id']).any():
        raise ValueError(f'{platform}: duplicate user-game interactions found')


# Keep users with enough known games for an observed/held-out split.
def eligible_users(df: pd.DataFrame, min_games: int) -> pd.DataFrame:
    count = df.groupby('user_id')['user_id'].transform('size')
    return df.loc[count >= min_games].copy()

# Hide part of each user's games, and always leave at least one observed game.
def split_user_libraries(
    df: pd.DataFrame, seed: int, heldout_game_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Check if the data is empty
    if df.empty:
        raise ValueError('No users have enough games for evaluation')

    # Sorting makes the split independent of the parquet row order.
    shuffled = (
        df.sort_values(['user_id', 'unified_game_id'])
        .sample(frac=1, random_state=seed)
    )
    positions = shuffled.groupby('user_id').cumcount()
    sizes = shuffled.groupby('user_id')['user_id'].transform('size')
    num_heldout = np.ceil(sizes * heldout_game_ratio).astype(int)
    num_heldout = num_heldout.clip(lower=1)
    heldout_mask = positions < num_heldout

    # Split the data into observed and heldout
    observed = shuffled.loc[~heldout_mask].sort_values(['user_id', 'unified_game_id'])
    heldout = shuffled.loc[heldout_mask].sort_values(['user_id', 'unified_game_id'])
    return observed.reset_index(drop=True), heldout.reset_index(drop=True)


def describe_split(observed: pd.DataFrame, heldout: pd.DataFrame) -> dict:
    # Check if the observed and heldout users are the same
    observed_ids = set(observed['user_id'])
    heldout_ids = set(heldout['user_id'])
    if observed_ids != heldout_ids:
        raise ValueError('Some evaluation users have no observed or held-out games')
    return {
        'users': len(observed_ids),
        'observed_interactions': len(observed),
        'heldout_interactions': len(heldout),
    }


def main() -> None:
    source = Path('data/interim')
    out = Path('data/processed')
    seed = 42
    steam_validation_user_ratio = 0.20
    heldout_game_ratio = 0.20
    min_games = 5

    steam = pd.read_parquet(source / 'steam_clean.parquet')
    ps = pd.read_parquet(source / 'ps_clean.parquet')
    check_interactions(steam, 'Steam')
    check_interactions(ps, 'PlayStation')

    # Split Steam by user, not by interaction, so validation users stay unseen.
    steam_eligible = eligible_users(steam, min_games)
    possible_val_users = np.sort(steam_eligible['user_id'].unique())
    if len(possible_val_users) < 2:
        raise ValueError('Not enough eligible Steam users for validation')
    n_val = max(
        1,
        min(
            len(possible_val_users) - 1,
            round(len(possible_val_users) * steam_validation_user_ratio),
        ),
    )
    rng = np.random.default_rng(seed)
    val_users = set(rng.choice(possible_val_users, size=n_val, replace=False))

    # Check if the user is in the validation set
    is_val = steam['user_id'].isin(val_users)

    # Split the data into train and validation
    train = steam.loc[~is_val].copy()
    validation = steam.loc[is_val].copy()

    # A model cannot rank games that never appear in training.
    # Keep only unified game IDs that appear in the training data.
    # Users with fewer than min_games will be excluded from evaluation.
    candidate_games = set(train['unified_game_id'])
    validation = eligible_users(
        validation[validation['unified_game_id'].isin(candidate_games)], min_games,
    )
    ps_test = eligible_users(
        ps[ps['unified_game_id'].isin(candidate_games)], min_games,
    )

    # Evaluation inputs are binary ownership. Playtime stays in the training file.
    validation = validation[['user_id', 'unified_game_id']]
    ps_test = ps_test[['user_id', 'unified_game_id']]

    # Split the data into observed and heldout for Steam
    steam_val_observed, steam_val_heldout = split_user_libraries(
        validation, seed + 1, heldout_game_ratio,
    )
    ps_test_observed, ps_test_heldout = split_user_libraries(
        ps_test, seed + 2, heldout_game_ratio,
    )

    if not set(train['user_id']).isdisjoint(validation['user_id']):
        raise ValueError('Steam train/validation user leakage')

    out.mkdir(parents=True, exist_ok=True)
    outputs = {
        'steam_train.parquet': train,
        'steam_val_observed.parquet': steam_val_observed,
        'steam_val_heldout.parquet': steam_val_heldout,
        'ps_test_observed.parquet': ps_test_observed,
        'ps_test_heldout.parquet': ps_test_heldout,
    }
    for filename, frame in outputs.items():
        frame.to_parquet(out / filename, index=False)

    report = {
        'seed': seed,
        'steam_validation_user_ratio': steam_validation_user_ratio,
        'heldout_game_ratio': heldout_game_ratio,
        'minimum_games_per_evaluation_user': min_games,
        'candidate_games_learned_from_steam_train': len(candidate_games),
        'steam_train': {
            'users': int(train['user_id'].nunique()),
            'interactions': len(train),
            'games': int(train['unified_game_id'].nunique()),
        },
        'steam_validation': describe_split(steam_val_observed, steam_val_heldout),
        'playstation_test': describe_split(ps_test_observed, ps_test_heldout),
        'ps_users_excluded_for_insufficient_shared_games': int(
            ps['user_id'].nunique() - ps_test['user_id'].nunique()
        ),
        'ps_interactions_not_in_steam_train_catalog': int(
            (~ps['unified_game_id'].isin(candidate_games)).sum()
        ),
    }
    (out / 'split_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8',
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print('\nSaved: data/processed/steam_train.parquet')
    print('Saved: data/processed/steam_val_observed.parquet')
    print('Saved: data/processed/steam_val_heldout.parquet')
    print('Saved: data/processed/ps_test_observed.parquet')
    print('Saved: data/processed/ps_test_heldout.parquet')
    print('Saved: data/processed/split_report.json')


if __name__ == '__main__':
    main()
