"""
Export selected parquet files to CSV.

Inputs:  the parquet paths listed in main
Outputs: one CSV per parquet, written into the output directory in main
"""

from pathlib import Path

import pandas as pd


def export_parquet(path: Path, out: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f'Missing parquet file: {path.as_posix()}')
    frame = pd.read_parquet(path)
    csv_path = out / f'{path.stem}.csv'
    frame.to_csv(csv_path, index=False)
    return csv_path


def main() -> None:
    inputs = (
        Path('data/processed/steam_train.parquet'),
        Path('data/processed/steam_val_observed.parquet'),
        Path('data/processed/steam_val_heldout.parquet'),
        Path('data/processed/ps_test_observed.parquet'),
        Path('data/processed/ps_test_heldout.parquet'),
    )
    out = Path('data/csv')

    out.mkdir(parents=True, exist_ok=True)
    for path in inputs:
        csv_path = export_parquet(path, out)
        print(f'Saved: {csv_path.as_posix()}')


if __name__ == '__main__':
    main()
