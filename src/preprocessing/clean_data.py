"""
Step 03: clean Steam / PlayStation interactions using shared game IDs.

Inputs:  data/raw/steam/*/users_games.csv, data/raw/playstation/purchased_games.csv,
         data/interim/shared_games.parquet
Outputs: data/interim/steam_clean.parquet, data/interim/ps_clean.parquet,
         data/interim/cleaning_report.json
"""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Keep shared games and collapse duplicate user-game rows. Playtime is Steam minutes.
def clean_steam(raw: Path, out: Path, mapping: dict[str, str]) -> dict:
    paths = sorted((raw / 'steam').glob('*/users_games.csv'))
    if not paths:
        raise FileNotFoundError('Missing data/raw/steam/*/users_games.csv')

    pieces = []
    stats = {'input_rows': 0, 'matched_rows': 0, 'invalid_playtimes_replaced': 0}
    cols = ['user_id', 'game_id', 'playtime_forever', 'playtime_2weeks']

    for path in paths:
        for chunk in pd.read_csv(
            path, usecols=cols, dtype='string', encoding='utf-8-sig', chunksize=200_000,
        ):
            # Count the number of rows in the Steam data
            stats['input_rows'] += len(chunk)

            # Fill empty values with empty strings and strip whitespace
            chunk['user_id'] = chunk['user_id'].fillna('').str.strip()
            chunk['game_id'] = chunk['game_id'].fillna('').str.strip()
            chunk = chunk.loc[chunk['user_id'].ne('')].copy()
            chunk['unified_game_id'] = chunk['game_id'].map(mapping)
            chunk = chunk.dropna(subset=['unified_game_id']).copy()

            # Count the number of rows in the matched data
            stats['matched_rows'] += len(chunk)

            # Non-numeric, infinite, and negative playtimes become 0 minutes.
            for col in ['playtime_forever', 'playtime_2weeks']:
                values = pd.to_numeric(chunk[col], errors='coerce')
                invalid = values.isna() | ~np.isfinite(values) | values.lt(0)
                stats['invalid_playtimes_replaced'] += int(invalid.sum())
                chunk[col] = values.mask(invalid, 0).astype('float64')
            pieces.append(chunk[['user_id', 'unified_game_id', 'playtime_forever', 'playtime_2weeks']])

    if not pieces:
        raise ValueError('No Steam interactions matched the shared games.')

    steam = pd.concat(pieces, ignore_index=True)
    # Playtime is a snapshot, so sum() would double-count duplicate records.
    steam = steam.groupby(['user_id', 'unified_game_id'], as_index=False, sort=False).agg({
        'playtime_forever': 'max', 'playtime_2weeks': 'max',
    })
    # Save the cleaned Steam data to a Parquet file
    steam.to_parquet(out / 'steam_clean.parquet', index=False)
    stats.update({
        'clean_rows': len(steam),
        'duplicate_pairs_collapsed': stats['matched_rows'] - len(steam),
        'unique_users': int(steam['user_id'].nunique()),
        'unique_games': int(steam['unified_game_id'].nunique()),
        'zero_forever_playtime_rows': int(steam['playtime_forever'].eq(0).sum()),
    })
    # Return the statistics for the cleaned Steam data
    return stats


# Expand each PlayStation library into user-game rows. Ownership records stay unfiltered.
def clean_playstation(raw: Path, out: Path, mapping: dict[str, str]) -> dict:
    path = raw / 'playstation' / 'purchased_games.csv'
    users = pd.read_csv(path, usecols=['playerid', 'library'], dtype='string', encoding='utf-8-sig')
    interactions = []
    stats = {
        'input_users': len(users),
        'unparsed_libraries': 0,
        'empty_user_ids': 0,
        'owned_ids_without_shared_match': 0,
        'duplicate_mapped_ids_collapsed': 0,
    }

    for player_id, library in users.itertuples(index=False, name=None):
        if pd.isna(player_id) or not str(player_id).strip():
            stats['empty_user_ids'] += 1
            continue
        try:
            ids = ast.literal_eval(str(library))
            if not isinstance(ids, list):
                raise ValueError('library must be a list')
        except (ValueError, SyntaxError, TypeError):
            stats['unparsed_libraries'] += 1
            continue

        mapped = [
            mapping[str(game_id).strip()]
            for game_id in ids if str(game_id).strip() in mapping
        ]
        stats['owned_ids_without_shared_match'] += len(ids) - len(mapped)
        unique_games = set(mapped)
        stats['duplicate_mapped_ids_collapsed'] += len(mapped) - len(unique_games)
        interactions.extend((str(player_id).strip(), game_id) for game_id in unique_games)

    ps = pd.DataFrame(interactions, columns=['user_id', 'unified_game_id'])
    ps = ps.drop_duplicates(['user_id', 'unified_game_id'])
    ps.to_parquet(out / 'ps_clean.parquet', index=False)
    stats.update({
        'clean_rows': len(ps),
        'unique_users': int(ps['user_id'].nunique()),
        'unique_games': int(ps['unified_game_id'].nunique()),
        'users_with_at_least_5_games': int(ps.groupby('user_id').size().ge(5).sum()),
    })
    return stats


def main() -> None:
    raw = Path('data/raw')
    out = Path('data/interim')
    out.mkdir(parents=True, exist_ok=True)

    shared = pd.read_parquet(out / 'shared_games.parquet')
    # One platform game ID must point at one unified ID.
    if (
        shared['ps_game_id'].duplicated().any()
        or shared.groupby('steam_game_id')['unified_game_id'].nunique().gt(1).any()
    ):
        raise ValueError('Shared game mapping is not one-to-one per platform game ID.')

    steam_map = dict(zip(shared['steam_game_id'].astype(str), shared['unified_game_id']))
    ps_map = dict(zip(shared['ps_game_id'].astype(str), shared['unified_game_id']))
    report = {
        'steam': clean_steam(raw, out, steam_map),
        'playstation': clean_playstation(raw, out, ps_map),
    }
    (out / 'cleaning_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8',
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print('\nSaved: data/interim/steam_clean.parquet')
    print('Saved: data/interim/ps_clean.parquet')
    print('Saved: data/interim/cleaning_report.json')


if __name__ == '__main__':
    main()
