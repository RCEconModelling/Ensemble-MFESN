"""
B=1 MFESN Tables: Medium MD and Small MD

Recompute Tables 2 and I.3 from the saved one-step-ahead forecasts.
Uses the same combination functions and settings as the main scripts.
No models are fitted and no forecast archives are overwritten.
"""

import argparse
import csv
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

# Load custom toolboxes
from comb_multistep_funcs import (
    median_forecast,
    simple_average_forecast,
    rollMSE_weights_forecast,
    ftl_forecast,
    const_hedge_forecast,
    dec_hedge_forecast,
    adahedge_forecast,
)

#region // GLOBAL SETTINGS ------------------------------------

PATH_WD = Path(__file__).resolve().parents[1]
PATH_TABLES = PATH_WD / 'tables_rev1'

DATASETS = [
    ('results_rev1', 'M', 'Table2_medium_MD.csv'),
    ('results_small_rev1', 'S', 'TableI3_small_MD.csv'),
]
ENSEMBLES = [
    ('RP', 'seed'),
    ('alphaRP', 'seed_leak'),
    ('lambdaRP', 'seed_lambda'),
    ('alpha-lambdaRP', 'seed_leak_lambda'),
]
MODELS = [
    ('S-MFESN A', 'singleESN_A'),
    ('S-MFESN B', 'singleESN_B'),
    ('M-MFESN A', 'multiESN_A'),
    ('M-MFESN B', 'multiESN_B'),
]
COLUMNS = [
    'Ensemble', 'Model', 'Method', 'MSFE', 'Relative_MSFE',
    'Change_vs_mean_pct', 'Combination_rank',
]

#endregion

#region --- Utility Functions

def load_results(path):
    # Only load the trusted result files supplied with this repository.
    with path.open('rb') as _file:
        return pickle.load(_file)


def check_forecasts(forecasts, targets, experts):
    """Check the full h=1 sample, allowing date labels within a quarter."""
    if forecasts.shape != (48, experts):
        raise ValueError(f'Expected 48 quarters and {experts} forecast columns.')
    if not forecasts.index.to_period('Q').equals(targets.index.to_period('Q')):
        raise ValueError('Forecasts and GDP targets must cover the same quarters.')
    if not np.isfinite(forecasts.to_numpy(dtype=float)).all():
        raise ValueError('Forecasts must be finite.')


def forecast_msfe(forecasts, targets):
    # With h=1, the main scripts use all 48 squared forecast errors.
    return float(np.mean(np.square(targets.to_numpy() - forecasts.to_numpy())))


def combination_forecasts(forecasts, losses, targets):
    """Use the original helpers; retain the main scripts' raw-loss settings."""
    combined = {}
    combined['Median'], _ = median_forecast(forecasts)
    combined['SA'], _ = simple_average_forecast(forecasts)
    combined['RollMSE'], _ = rollMSE_weights_forecast(
        forecasts, targets=targets, h=1, window_size=4,
    )
    combined['FTL'], _ = ftl_forecast(forecasts, losses=losses, h=1)
    combined['Hedge'], _ = const_hedge_forecast(
        forecasts, losses=losses, h=1, learning_rate=1e4,
    )
    combined['DecHedge'], _ = dec_hedge_forecast(
        forecasts, losses=losses, h=1, c0=1e4,
    )
    combined['AdaHedge'], _ = adahedge_forecast(forecasts, losses=losses, h=1)
    return combined


def table_row(ensemble, model, method, msfe, reference_msfe, rank=''):
    relative_msfe = msfe / reference_msfe
    # Manuscript percentages are relative to the in-sample mean, not the
    # single-draw MFESN baseline used in the main scripts' console percentages.
    return [
        ensemble, model, method, msfe, relative_msfe,
        100 * (relative_msfe - 1), rank,
    ]

#endregion

#region --- Construct Tables

def results_table(results_folder, dataset):
    path_results = PATH_WD / results_folder
    dataset2007 = load_results(path_results / 'dataset2007_GDP_data.pkl')
    targets = dataset2007['GDP_data_test']
    expected_quarters = pd.period_range('2008Q1', '2019Q4', freq='Q')
    if targets.shape != (48, 1) or not targets.index.to_period('Q').equals(expected_quarters):
        raise ValueError('Expected GDP targets for all 48 quarters, 2008Q1-2019Q4.')
    if not np.isfinite(targets.to_numpy()).all():
        raise ValueError('GDP targets must be finite.')

    # Saved tuple: Mean, AR(1), DFM A/B, S-MFESN A/B, M-MFESN A/B.
    fixed_forecasts = load_results(
        path_results / f'results_{dataset}__multistep_fixed_parameters.pkl'
    )
    if len(fixed_forecasts) != 8:
        raise ValueError('Expected eight fixed-parameter benchmark forecasts.')
    for forecasts in fixed_forecasts:
        check_forecasts(forecasts, targets, experts=1)
    fixed_msfe = [forecast_msfe(forecasts, targets) for forecasts in fixed_forecasts]
    reference_msfe = fixed_msfe[0]
    if reference_msfe <= 0:
        raise ValueError('The in-sample-mean forecast must have positive MSFE.')

    rows = []
    for name, msfe in zip(['Mean', 'AR(1)', 'DFM A', 'DFM B'], fixed_msfe[:4]):
        rows.append(table_row('Benchmark', name, 'Baseline', msfe, reference_msfe))

    for ensemble, suffix in ENSEMBLES:
        for model_index, (model, filename) in enumerate(MODELS):
            results = load_results(
                path_results / f'results_{dataset}__multistep_{filename}_fix_grid_{suffix}.pkl'
            )
            # The first two tuple entries contain forecasts and squared errors,
            # each with one list entry for h=1. No new reservoir draws are made.
            if len(results[0]) != 1 or len(results[1]) != 1:
                raise ValueError('Expected saved one-step-ahead ensemble results.')
            forecasts, losses = results[0][0], results[1][0]
            check_forecasts(forecasts, targets, experts=1000)
            if not forecasts.index.equals(targets.index):
                raise ValueError('Ensemble forecasts must use the saved GDP target index.')
            if losses.shape != forecasts.shape or not losses.index.equals(forecasts.index):
                raise ValueError('Saved losses must match the ensemble forecast shape and index.')
            np.testing.assert_allclose(
                losses.to_numpy(), np.square(targets.to_numpy() - forecasts.to_numpy()),
                rtol=1e-13, atol=0,
                err_msg='Saved combination losses must be unclipped squared errors.',
            )

            combined = combination_forecasts(forecasts, losses, targets)
            msfes = {}
            for method, combined_forecast in combined.items():
                check_forecasts(combined_forecast, targets, experts=1)
                msfes[method] = forecast_msfe(combined_forecast, targets)
            ranking = sorted(msfes, key=msfes.get)

            rows.append(table_row(
                ensemble, model, 'Baseline', fixed_msfe[model_index + 4], reference_msfe,
            ))
            for method, msfe in msfes.items():
                rows.append(table_row(
                    ensemble, model, method, msfe, reference_msfe, ranking.index(method) + 1,
                ))

    return rows


def main():
    parser = argparse.ArgumentParser(description='Export B=1 Tables 2 and I.3 from saved forecasts.')
    parser.add_argument('--output-dir', type=Path, default=PATH_TABLES,
                        help='CSV destination (default: tables_rev1 in the repository).')
    args = parser.parse_args()

    # Finish both calculations before writing the two table exports.
    tables = [(filename, results_table(folder, dataset)) for folder, dataset, filename in DATASETS]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in tables:
        path = args.output_dir / filename
        with path.open('w', encoding='utf-8', newline='') as _file:
            writer = csv.writer(_file)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        print(f'[!] Table saved as CSV ({len(rows)} rows):\n\t{path}')


if __name__ == '__main__':
    main()

#endregion
