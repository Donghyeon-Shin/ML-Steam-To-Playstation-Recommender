"""
Step 01: inspect the raw Steam / PlayStation CSVs.

Inputs:  data/raw/steam/*/games.csv, data/raw/steam/*/users_games.csv,
         data/raw/playstation/games.csv, data/raw/playstation/purchased_games.csv
Outputs: data/interim/raw_audit.json, data/interim/raw_audit.md

data/raw/archive is a different dump and is not part of this audit.
"""

from __future__ import annotations

import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Globs are relative to data/raw. Partition folders are found, not listed.
LAYOUT = (
    ('steam_games', 'steam/*/games.csv'),
    ('steam_user_games', 'steam/*/users_games.csv'),
    ('playstation_games', 'playstation/games.csv'),
    ('playstation_purchases', 'playstation/purchased_games.csv'),
)

def clean_text(col: pd.Series) -> pd.Series:
    return col.fillna('').astype('string').str.strip()

# Check for duplicate values in a column
def nonempty_duplicates(col: pd.Series) -> int:
    keys = clean_text(col)
    return int(keys[keys.ne('')].duplicated().sum())


# Check for unique values in a column
def nonempty_nunique(col: pd.Series) -> int:
    return int(clean_text(col).replace('', pd.NA).nunique())

# Parse a string into a list of strings
# Example: "['1234567890', '1234567891']" -> ['1234567890', '1234567891']
def parse_library(value: str) -> list[str] | None:
    text = value.strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, list):
        return None
    return [str(item).strip() for item in parsed]

# Read a sample of rows from a CSV file
def read_sample(path: Path, sample_rows: int) -> pd.DataFrame:
    # Identifiers stay strings so they are never rounded into floats.
    return pd.read_csv(
        path, nrows=sample_rows, dtype='string', encoding='utf-8-sig',
        keep_default_na=False, low_memory=False,
    )

# Discover the files in the data/raw directory
def discover(raw: Path) -> tuple[list[tuple[str, Path]], list[str]]:
    found: list[tuple[str, Path]] = []
    missing: list[str] = []
    for kind, pattern in LAYOUT: # Checking for the files in the data/raw directory
        matches = sorted(raw.glob(pattern))
        if not matches:
            missing.append(pattern)
        found.extend((kind, path) for path in matches)
    return found, missing


def inspect(path: Path, raw: Path, sample_rows: int, kind: str) -> dict:
    info = {
        'path': path.relative_to(raw).as_posix(),
        'kind': kind,
        'status': 'ok',
        'sample_limit': sample_rows,
    }
    try:
        df = read_sample(path, sample_rows) # Data read
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        info.update(status='error', error=f'{type(exc).__name__}: {exc}')
        return info

    cols = list(df.columns) # The columns in the data
    info['columns'] = cols
    info['rows_sampled'] = len(df)
    info['sample_covers_whole_file'] = len(df) < sample_rows
    missing = {col: int(clean_text(df[col]).eq('').sum()) for col in cols}
    info['empty_cells_in_sample'] = {col: value for col, value in missing.items() if value}

    checkers = {
        'steam_games': _steam_games,
        'steam_user_games': _steam_user_games,
        'playstation_games': _playstation_games,
        'playstation_purchases': _playstation_purchases,
    }
    error = checkers[kind](df, info)
    if error:
        info.update(status='error', error=error)
    return info


def _require(df: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    missing = [col for col in columns if col not in df.columns]
    if not missing:
        return None
    return f'missing columns {missing}; found {list(df.columns)}'


# Check for duplicate game IDs and titles in the steam_games file
def _steam_games(df: pd.DataFrame, info: dict) -> str | None:
    if error := _require(df, ('id', 'name')):
        return error
    info['duplicate_game_ids_in_sample'] = nonempty_duplicates(df['id'])
    info['unique_games_in_sample'] = nonempty_nunique(df['id'])
    titles = clean_text(df['name']).str.casefold()
    info['duplicate_casefold_titles_in_sample'] = int(titles[titles.ne('')].duplicated().sum())
    return None


# users_games.csv is one row per user-game. Playtime columns are Steam minutes.
def _steam_user_games(df: pd.DataFrame, info: dict) -> str | None:
    columns = ('user_id', 'game_id', 'playtime_2weeks', 'playtime_forever')
    if error := _require(df, columns):
        return error
    info['unique_users_in_sample'] = nonempty_nunique(df['user_id'])
    info['unique_games_in_sample'] = nonempty_nunique(df['game_id'])
    pairs = df[['user_id', 'game_id']].copy()
    pairs['user_id'] = clean_text(pairs['user_id'])
    pairs['game_id'] = clean_text(pairs['game_id'])
    valid = pairs['user_id'].ne('') & pairs['game_id'].ne('')
    info['duplicate_user_game_pairs_in_sample'] = int(pairs.loc[valid].duplicated().sum())
    info['playtime_minutes'] = {
        col: _playtime_minutes(df[col]) for col in ('playtime_2weeks', 'playtime_forever')
    }
    return None


def _playtime_minutes(col: pd.Series) -> dict:
    numbers = pd.to_numeric(clean_text(col), errors='coerce')
    valid = numbers.dropna()

    def summarize(fn):
        if valid.empty:
            return None
        value = float(fn())
        return int(value) if value.is_integer() else round(value, 1)

    # Return the statistics for the playtime minutes
    return {
        'non_numeric_or_empty': int(numbers.isna().sum()),
        'zero_values': int(valid.eq(0).sum()),
        'negative_values': int(valid.lt(0).sum()),
        'median': summarize(valid.median),
        'p99': summarize(lambda: valid.quantile(0.99)),
        'max': summarize(valid.max),
    }


# Check for duplicate game IDs and titles in the playstation_games file
def _playstation_games(df: pd.DataFrame, info: dict) -> str | None:
    if error := _require(df, ('gameid', 'title')):
        return error
    info['duplicate_game_ids_in_sample'] = nonempty_duplicates(df['gameid'])
    info['unique_games_in_sample'] = nonempty_nunique(df['gameid'])
    titles = clean_text(df['title']).str.casefold()
    info['duplicate_casefold_titles_in_sample'] = int(titles[titles.ne('')].duplicated().sum())
    return None


# Check for duplicate user IDs and library in the playstation_purchases file
def _playstation_purchases(df: pd.DataFrame, info: dict) -> str | None:
    if error := _require(df, ('playerid', 'library')):
        return error
    info['duplicate_user_ids_in_sample'] = nonempty_duplicates(df['playerid'])
    info['unique_users_in_sample'] = nonempty_nunique(df['playerid'])

    game_ids: set[str] = set()
    empty = unparsed = inner_dups = 0
    for raw in clean_text(df['library']):
        library = parse_library(raw)
        if library is None:
            unparsed += 1
            continue
        if not library:
            empty += 1
            continue
        inner_dups += len(library) - len(set(library))
        game_ids.update(item for item in library if item)
    info['unique_games_in_sample'] = len(game_ids)
    info['empty_libraries_in_sample'] = empty
    info['unparsed_libraries_in_sample'] = unparsed
    info['duplicate_ids_inside_libraries_in_sample'] = inner_dups
    return None

# The keys for the statistics in the report
STAT_KEYS = (
    'unique_users_in_sample',
    'unique_games_in_sample',
    'duplicate_game_ids_in_sample',
    'duplicate_user_ids_in_sample',
    'duplicate_user_game_pairs_in_sample',
    'duplicate_casefold_titles_in_sample',
    'empty_libraries_in_sample',
    'unparsed_libraries_in_sample',
    'duplicate_ids_inside_libraries_in_sample',
)

# Addtion markdown formatting to the report
def as_markdown(report: dict) -> str:
    lines = [
        '# Step 01 — Raw CSV Data Audit', '',
        f"Rows inspected per file: up to {report['sample_rows_per_file']:,}",
    ]
    for rec in report['files']:
        lines += ['', f"## `{rec['path']}`", '']
        if rec['status'] != 'ok':
            lines.append(f"- Error: {rec.get('error', 'unknown')}")
            continue
        lines += [
            f"- Kind: {rec['kind']}",
            f"- Sample rows: {rec['rows_sampled']:,}",
            f"- Sample covers whole file: {rec['sample_covers_whole_file']}",
            f"- Columns: `{', '.join(rec['columns'])}`",
            f"- Empty cells (sample): `{json.dumps(rec['empty_cells_in_sample'], ensure_ascii=False)}`",
        ]
        for key in STAT_KEYS:
            if key in rec:
                lines.append(f'- {key}: {rec[key]:,}')
        for col, stats in rec.get('playtime_minutes', {}).items():
            rendered = ', '.join(
                f'{key}={value:,}' if isinstance(value, int) else f'{key}={value}'
                for key, value in stats.items()
            )
            lines.append(f'- {col} (minutes): {rendered}')
    return '\n'.join(lines) + '\n'


def main() -> int:
    raw = Path('data/raw')
    out = Path('data/interim')
    sample_rows = 50000 # The maximum number of rows to sample from each file

    discovered, missing = discover(raw) # Discover the files in the data/raw directory
    files = [] # A list of dictionaries, one for each file

    for kind, path in discovered:
        result = inspect(path, raw, sample_rows, kind) # Inspect the file
        files.append(result) # Add the result to the list
        rel = result['path']

        # Check status of the inspection
        if result['status'] != 'ok':
            print(f"[ERROR] {rel} | {result.get('error', 'unknown')}")
            continue
        print(f"[OK] {rel} | rows sampled: {result['rows_sampled']:,} | "
              f"columns: {result['columns']}")

    for pattern in missing:
        print(f'[MISSING] {pattern}')

    # Report
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), # The date and time the report was created
        'sample_rows_per_file': sample_rows,
        'files_found': len(files),
        'missing_patterns': missing,
        'files': files,
    }

    # Save the report to a JSON and a markdown file
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / 'raw_audit.json'
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    
    md_path = out / 'raw_audit.md'
    md_path.write_text(as_markdown(report), encoding='utf-8')

    print(f'\nSaved: {json_path} and {md_path}')
    
    if missing or any(item['status'] != 'ok' for item in files):
        print('Some CSV files are missing or unreadable; check the report.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
