"""
Step 02: match Steam and PlayStation game catalogs by normalized title.

Inputs:  data/raw/steam/*/games.csv, data/raw/playstation/games.csv,
         data/raw/playstation/purchased_games.csv (coverage check only)
Outputs: data/interim/shared_games.parquet, game_matching_report.json
"""

import ast
import json
import re
import unicodedata
from pathlib import Path
from statistics import median

import pandas as pd


# Ignore case, punctuation, and repeated spaces. Keep edition names and numbers.
def normalize_title(value: str) -> str:
    text = unicodedata.normalize('NFKC', str(value)).casefold()
    text = re.sub(r'[\W_]+', ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).strip()


def read_catalogs(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    paths = sorted((raw / 'steam').glob('*/games.csv'))
    if not paths:
        raise FileNotFoundError('Missing data/raw/steam/*/games.csv')

    # Read the Steam games data
    steam = pd.concat(
        [pd.read_csv(p, usecols=['id', 'name'], dtype='string', encoding='utf-8-sig')
         for p in paths],
        ignore_index=True,
    ).rename(columns={'id': 'steam_game_id', 'name': 'steam_title'})

    # Read the PlayStation games data
    ps_path = raw / 'playstation' / 'games.csv'
    ps = pd.read_csv(
        ps_path, usecols=['gameid', 'title', 'platform'],
        dtype='string', encoding='utf-8-sig',
    ).rename(columns={
        'gameid': 'ps_game_id', 'title': 'ps_title', 'platform': 'ps_platform',
    })

    # Count the number of rows in the Steam and PlayStation data
    counts = {'steam_metadata_rows': len(steam), 'ps_metadata_rows': len(ps)}

    # Normalize the title of the games
    for df, id_col, title_col, platform in (
        (steam, 'steam_game_id', 'steam_title', 'steam'),
        (ps, 'ps_game_id', 'ps_title', 'ps'),
    ):
        # Fill empty values with empty strings and strip whitespace
        df[id_col] = df[id_col].fillna('').str.strip()
        df[title_col] = df[title_col].fillna('').str.strip()

        # Normalize the title of the games
        df['normalized_title'] = df[title_col].map(normalize_title)

        # Check & Counting if the ID or title is empty
        bad = df[id_col].eq('') | df['normalized_title'].eq('')
        counts[f'{platform}_rows_missing_id_or_title'] = int(bad.sum()) 

        # Drop the rows with empty ID or title
        df.drop(df.index[bad], inplace=True)

        # A platform ID linked to different titles cannot be matched reliably.
        inconsistent_ids = df.groupby(id_col)['normalized_title'].nunique()
        inconsistent_ids = set(inconsistent_ids[inconsistent_ids > 1].index)
        counts[f'{platform}_ids_with_conflicting_titles'] = len(inconsistent_ids)
        df.drop(df.index[df[id_col].isin(inconsistent_ids)], inplace=True)
        df.drop_duplicates(subset=id_col, keep='first', inplace=True)

    # Different Steam IDs with the same title are ambiguous: exclude them.
    steam_title_counts = steam.groupby('normalized_title')['steam_game_id'].nunique()
    ambiguous_titles = set(steam_title_counts[steam_title_counts > 1].index)
    counts['ambiguous_steam_titles_excluded'] = len(ambiguous_titles)
    steam = steam.loc[~steam['normalized_title'].isin(ambiguous_titles)].copy()

    # PlayStation may have multiple platform-specific IDs for the same game.
    counts['ps_titles_with_multiple_ids'] = int(
        ps.groupby('normalized_title')['ps_game_id'].nunique().gt(1).sum()
    )
    counts['steam_games_available_for_matching'] = len(steam)
    counts['ps_games_available_for_matching'] = len(ps)
    return steam, ps, counts

# Audit the purchases of the PlayStation games
def audit_purchases(raw: Path, ps_ids: set[str], ps_to_unified: dict[str, str]) -> dict:
    path = raw / 'playstation' / 'purchased_games.csv'
    owned_ids: set[str] = set()
    shared_counts: list[int] = []
    invalid_libraries = 0

    for chunk in pd.read_csv(path, usecols=['library'], dtype='string',
                             encoding='utf-8-sig', chunksize=1000):
        # Read the purchased games data in chunks of 1000 rows
        for raw_library in chunk['library']:
            try:
                values = ast.literal_eval(str(raw_library))
                if not isinstance(values, list):
                    raise ValueError('library is not a list')
            except (ValueError, SyntaxError, TypeError):
                invalid_libraries += 1
                shared_counts.append(0)
                continue
            ids = {str(value).strip() for value in values if str(value).strip()}
            owned_ids.update(ids)
            shared_counts.append(len({ps_to_unified[g] for g in ids if g in ps_to_unified}))

    unknown = owned_ids - ps_ids
    # Return the statistics for the purchased games
    return {
        'users_checked': len(shared_counts),
        'unique_owned_ps_game_ids': len(owned_ids),
        'owned_ps_ids_missing_from_catalog': len(unknown),
        'missing_ps_id_examples': sorted(unknown)[:10],
        'invalid_libraries': invalid_libraries,
        'users_with_at_least_2_shared_games': sum(n >= 2 for n in shared_counts),
        'users_with_at_least_5_shared_games': sum(n >= 5 for n in shared_counts),
        'users_with_at_least_10_shared_games': sum(n >= 10 for n in shared_counts),
        'median_shared_games_per_user': median(shared_counts) if shared_counts else 0,
    }


def main() -> None:
    raw = Path('data/raw')
    out = Path('data/interim')
    steam, ps, report = read_catalogs(raw)
    shared = ps.merge(steam, on='normalized_title', how='inner', validate='many_to_one')

    # Deterministic unified ID per title. Several PS IDs can map to one ID.
    titles = sorted(shared['normalized_title'].unique())
    unified_ids = {title: f'G{i:06d}' for i, title in enumerate(titles, start=1)}
    shared.insert(0, 'unified_game_id', shared['normalized_title'].map(unified_ids))

    # Sort the shared data by unified game ID and PlayStation game ID
    shared = shared[[
        'unified_game_id', 'normalized_title', 'steam_game_id', 'steam_title',
        'ps_game_id', 'ps_title', 'ps_platform',
    ]].sort_values(['unified_game_id', 'ps_game_id']).reset_index(drop=True)

    # Update the report with the statistics for the shared games
    report.update({
        'shared_unique_titles': len(titles),
        'matched_steam_game_ids': int(shared['steam_game_id'].nunique()),
        'matched_ps_game_ids': int(shared['ps_game_id'].nunique()),
        'mapping_rows': len(shared),
        'ps_platform_counts': ps['ps_platform'].fillna('(missing)').value_counts().to_dict(),
    })
    report['ps_purchase_coverage'] = audit_purchases(
        raw,
        set(ps['ps_game_id']),
        dict(zip(shared['ps_game_id'], shared['unified_game_id'])),
    )

    out.mkdir(parents=True, exist_ok=True)
    shared.to_parquet(out / 'shared_games.parquet', index=False)
    (out / 'game_matching_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8',
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print('\nSaved: data/interim/shared_games.parquet')
    print('Saved: data/interim/game_matching_report.json')
    if report['ps_purchase_coverage']['owned_ps_ids_missing_from_catalog']:
        print('\nWARNING: Some purchased PlayStation game IDs have no catalog metadata.')
        print('They are NOT mapped; check coverage before final evaluation.')


if __name__ == '__main__':
    main()
