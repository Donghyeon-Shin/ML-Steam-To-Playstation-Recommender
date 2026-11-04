"""Create Item-Based-only comparison tables and HTML heatmaps (no extra packages)."""
import argparse
from html import escape
from pathlib import Path
import pandas as pd


def create_report(directory):
    directory = Path(directory)
    df = pd.read_csv(directory / 'grid_results.csv')
    score = 'ndcg@10'
    ranked = df.sort_values(score, ascending=False, kind='stable')
    ranked.to_csv(directory / 'ranked_results.csv', index=False)
    comparisons = []
    for mode in ['binary', 'playtime']:
        reference = df[(df.stage == 'reference') & (df.weighting == mode)].iloc[0]
        candidates = ranked[(ranked.stage != 'reference') & (ranked.weighting == mode)]
        if candidates.empty:
            continue
        best = candidates.iloc[0]
        comparisons.append({'weighting': mode, 'reference_ndcg@10': reference[score],
            'best_tuned_ndcg@10': best[score], 'difference': best[score] - reference[score],
            'relative_change_percent': 100 * (best[score] / reference[score] - 1)
                if reference[score] else None,
            'best_trial': best.trial, 'neighbors': best.neighbors, 'shrinkage': best.shrinkage,
            'beta': best.get('beta'), 'cap_quantile': best.get('cap_quantile'),
            'min_common_users': best.get('min_common_users'),
            'min_similarity': best.get('min_similarity')})
    summary = pd.DataFrame(comparisons)
    summary.to_csv(directory / 'before_after.csv', index=False)
    sections = ['<h1>Item-Based CF: Steam validation</h1>',
                '<p>Selection: maximum mean NDCG@10. Staged search, not exhaustive joint search. '
                'Independent scales per heatmap. Missing cells were not evaluated.</p>',
                '<h2>Reference vs tuned best</h2>', summary.to_html(index=False, float_format=lambda x:f'{x:.6f}')]
    for stage, row_key, col_key in [('binary_grid','neighbors','shrinkage'),
                                    ('binary_filters','min_common_users','min_similarity'),
                                    ('playtime_grid','beta','cap_quantile'),
                                    ('playtime_refine','neighbors','shrinkage'),
                                    ('playtime_filters','min_common_users','min_similarity')]:
        part = df[df.stage == stage].copy()
        if part.empty:
            continue
        if col_key == 'cap_quantile':
            part[col_key] = part[col_key].map(lambda v: 'none' if pd.isna(v) else str(v))
        pivot = part.pivot_table(index=row_key, columns=col_key, values=score, aggfunc='first')
        pivot.to_csv(directory / f'{stage}_ndcg.csv')
        low, high = part[score].min(), part[score].max()
        sections.append(f'<h2>{stage}: {row_key} × {col_key}</h2><table><tr><th>{row_key}</th>')
        sections.extend(f'<th>{escape(str(c))}</th>' for c in pivot.columns)
        sections.append('</tr>')
        for label, row in pivot.iterrows():
            sections.append(f'<tr><th>{escape(str(label))}</th>')
            for value in row:
                if pd.isna(value):
                    sections.append('<td>—</td>')
                else:
                    strength = 0.5 if high == low else (value-low)/(high-low)
                    color = int(245 - 100 * strength)
                    sections.append(f'<td style="background:rgb({color},235,{color})">{value:.6f}</td>')
            sections.append('</tr>')
        sections.append('</table>')
    sections.extend(['<h2>All trials, descending NDCG@10</h2>',
                     ranked.to_html(index=False, float_format=lambda x:f'{x:.6f}')])
    html = '<!doctype html><meta charset="utf-8"><title>Item-Based tuning</title><style>' \
           'body{font-family:system-ui;margin:32px}table{border-collapse:collapse;margin-bottom:24px}' \
           'th,td{border:1px solid #ccc;padding:8px;text-align:right}th{background:#eee}' \
           '</style>' + ''.join(sections)
    (directory / 'comparison.html').write_text(html, encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=Path('results/item_based/tuning_v1'))
    create_report(parser.parse_args().results_dir)
