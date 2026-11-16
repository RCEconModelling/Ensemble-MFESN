"""
B=100 MFESN Tables: Medium MD

Table I.5: ratios of mean MSFEs across the 100 pools.
Table I.6: CV-selected expert ranks and paired FTL/AdaHedge win rates.
Uses compact saved results; no models are fitted or combinations reweighted.
"""

import argparse
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

#region // GLOBAL SETTINGS ------------------------------------

PATH_WD = Path(__file__).resolve().parents[1]
MODELS = [('S-MFESN A', 'smfesn_a'), ('S-MFESN B', 'smfesn_b'),
          ('M-MFESN A', 'mmfesn_a'), ('M-MFESN B', 'mmfesn_b')]
ENSEMBLES = ['RP', 'alphaRP', 'lambdaRP', 'alpha-lambdaRP']
METHODS = ['Median', 'SA', 'RollMSE', 'FTL', 'Hedge', 'DecHedge', 'AdaHedge']

#endregion

#region --- Construct Tables

def check_results(data):
    expected = {'cv_scores': (100, 1000), 'expert_msfe': (100, 1000),
                'ridge_penalties': (100, 1000), 'selected_ids': (100,),
                'selected_forecasts': (100, 48), 'combined_forecasts': (100, 48, 7),
                'observed': (48,), 'cv_msfe': (100,), 'method_msfe': (100, 7)}
    for name, shape in expected.items():
        if data[name].shape != shape or not np.isfinite(data[name]).all():
            raise ValueError(f'Invalid {name}: expected {shape}, all finite.')
    if data['methods'].tolist() != METHODS:
        raise ValueError('Unexpected combination order.')
    if data['quarters'].tolist() != pd.period_range('2008Q1', '2019Q4', freq='Q').astype(str).tolist():
        raise ValueError('Expected all 48 quarters, 2008Q1-2019Q4.')
    np.testing.assert_array_equal(data['selected_ids'], np.argmin(data['cv_scores'], axis=1))
    # Preserve saved loss reductions; independently check against saved paths.
    cv_loss = np.mean((data['selected_forecasts'] - data['observed'])**2, axis=1)
    method_loss = np.mean((data['combined_forecasts'] - data['observed'][None,:,None])**2, axis=1)
    np.testing.assert_allclose(cv_loss, data['cv_msfe'], rtol=2e-11, atol=3e-18)
    np.testing.assert_allclose(method_loss, data['method_msfe'], rtol=2e-11, atol=3e-18)
    np.testing.assert_allclose(data['expert_msfe'][np.arange(100), data['selected_ids']], cv_loss,
                               rtol=2e-11, atol=3e-18)
    if float(data['benchmark_msfe']) <= 0:
        raise ValueError('In-sample-mean benchmark MSFE must be positive.')


def rank_display(value):
    # Display ranks as integers; retain unrounded quartiles in separate columns.
    return str(Decimal(str(value)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def make_tables(path_results, path_tables):
    manifest = Path(path_results) / 'manifest.json'
    if manifest.exists():
        for name, expected in json.loads(manifest.read_text())['files'].items():
            actual = hashlib.sha256((Path(path_results)/name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'Saved result hash differs: {name}')
    msfe_rows, rank_rows, pool_rows = [], [], []
    for model, tag in MODELS:
        for ensemble in ENSEMBLES:
            with np.load(Path(path_results) / f'{tag}_{ensemble}.npz', allow_pickle=False) as data:
                check_results(data)
                cv = data['cv_msfe']
                means = data['method_msfe'].mean(axis=0)
                ratios = means / cv.mean()
                order = np.argsort(means, kind='stable')
                row = dict(Model=model, Ensemble=ensemble, CV=cv.mean()/float(data['benchmark_msfe']))
                row.update(zip(METHODS, ratios))
                row.update(Best=METHODS[order[0]], Second_best=METHODS[order[1]])
                msfe_rows.append(row)

                selected_loss = data['expert_msfe'][np.arange(100), data['selected_ids']]
                ranks = 1 + np.sum(data['expert_msfe'] < selected_loss[:,None], axis=1)
                q1, median, q3 = np.quantile(ranks, [.25, .5, .75], method='linear')
                differences = data['method_msfe'][:, [METHODS.index('FTL'), METHODS.index('AdaHedge')]] - cv[:,None]
                wins = 100 * np.mean(differences < 0, axis=0)
                rank_rows.append(dict(Model=model, Ensemble=ensemble,
                    CV_OOS_rank=f'{rank_display(median)} [{rank_display(q1)}, {rank_display(q3)}]',
                    Median_rank=median, Q1_rank=q1, Q3_rank=q3,
                    FTL_wins_pct=int(round(wins[0])), AdaHedge_wins_pct=int(round(wins[1]))))
                for b in range(100):
                    pool_rows.append(dict(Model=model, Ensemble=ensemble, Replication=b+1,
                        CV_expert_index=int(data['selected_ids'][b]), CV_MSFE=cv[b], CV_OOS_rank=int(ranks[b]),
                        FTL_minus_CV=differences[b,0], AdaHedge_minus_CV=differences[b,1]))
    path_tables = Path(path_tables)
    path_tables.mkdir(parents=True, exist_ok=True)
    for filename, rows in [('TableI5_medium_MD_B100.csv', msfe_rows),
                           ('TableI6_medium_MD_B100_ranks_wins.csv', rank_rows),
                           ('medium_MD_B100_paired_losses.csv', pool_rows)]:
        pd.DataFrame(rows).to_csv(path_tables / filename, index=False, float_format='%.17g')
        print(f'Saved {path_tables / filename}')
    return pd.DataFrame(msfe_rows), pd.DataFrame(rank_rows)

#endregion

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=PATH_WD / 'results_b100_rev1')
    parser.add_argument('--tables', type=Path, default=PATH_WD / 'tables_rev1')
    args = parser.parse_args()
    make_tables(args.results, args.tables)
