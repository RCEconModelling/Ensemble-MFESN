"""
Ensemble MFESN Medium MD: Quantile-Clipped Combination Losses

B=1, h=1; Tables I.1 and I.2.
Reuses the saved expert forecasts. Only the online combination loss and
the corresponding Hedge learning-rate parameters change from Table 2.
"""

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

# Load custom toolboxes
from comb_multistep_funcs import (
    ftl_forecast,
    const_hedge_forecast,
    dec_hedge_forecast,
    adahedge_forecast,
)

#region // GLOBAL SETTINGS ------------------------------------

FORECAST_STEPS = 1
ESN_MATRIX_RESAMPLES = 1000
CLIPPING_PERCENTILES = [95, 99]
PAIR_RESAMPLES = 20000
PAIR_SEED = 20261001
ETA = 1e4
C0 = 1e4

SAVE_RESULTS = True
SAVE_CSVS = True

PATH_WD = Path(__file__).resolve().parents[1]
PATH_RESULTS = PATH_WD / 'results_rev1'
PATH_CLIPPED = PATH_WD / 'results_clipped_rev1'
PATH_TABLES = PATH_WD / 'tables_rev1'

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
METHODS = ['FTL', 'Hedge', 'DecHedge', 'AdaHedge']
console = Console(width=120)

#endregion

#region --- Utility Functions for Forecast Combinations

def load_results(path):
    # Only load the trusted result files supplied with this repository.
    with path.open('rb') as _file:
        return pickle.load(_file)


def combine_forecasts(forecasts, losses, eta, c0):
    # Same corrected routines as ensemble_multistep_medium_md.py.
    ftl, _ = ftl_forecast(forecasts, losses=losses, h=FORECAST_STEPS)
    hedge, _ = const_hedge_forecast(
        forecasts, losses=losses, h=FORECAST_STEPS, learning_rate=eta,
    )
    dechedge, _ = dec_hedge_forecast(
        forecasts, losses=losses, h=FORECAST_STEPS, c0=c0,
    )
    adahedge, _ = adahedge_forecast(forecasts, losses=losses, h=FORECAST_STEPS)
    combined = pd.concat([ftl, hedge, dechedge, adahedge], axis=1)
    combined.columns = METHODS
    if not np.isfinite(combined.to_numpy()).all():
        raise ValueError('Combination forecasts must be finite.')
    return combined


def evaluation_losses(forecasts, targets):
    # Evaluation always uses unclipped squared errors, including for 95/Raw
    # and 99/Raw. Columns retain the forecast or expert names.
    return pd.DataFrame(
        np.square(forecasts.to_numpy() - targets.to_numpy()),
        index=forecasts.index, columns=forecasts.columns,
    )

#endregion

def main():
    #region --- Load Saved Forecast Inputs

    dataset2007 = load_results(PATH_RESULTS / 'dataset2007_GDP_data.pkl')
    training = dataset2007['GDP_data_train']
    targets = dataset2007['GDP_data_test']
    assert FORECAST_STEPS == 1
    assert training.shape == (72, 1) and targets.shape == (48, 1)
    assert training.index.to_period('Q').equals(pd.period_range('1990Q1', '2007Q4', freq='Q'))
    assert targets.index.to_period('Q').equals(pd.period_range('2008Q1', '2019Q4', freq='Q'))
    assert np.isfinite(training.to_numpy()).all() and np.isfinite(targets.to_numpy()).all()

    fixed_forecasts = load_results(PATH_RESULTS / 'results_M__multistep_fixed_parameters.pkl')
    mean_forecast = fixed_forecasts[0]
    np.testing.assert_array_equal(mean_forecast.to_numpy(), np.full((48, 1), training.to_numpy().mean()))
    reference_msfe = float(evaluation_losses(mean_forecast, targets).to_numpy().mean())

    #endregion

    #region --- Calibrate Clipping Thresholds from Training Observations

    train = training.iloc[:, 0].to_numpy()
    rng = np.random.default_rng(PAIR_SEED)
    left_indices = rng.integers(0, len(train), size=PAIR_RESAMPLES)
    right_indices = rng.integers(0, len(train), size=PAIR_RESAMPLES)
    squared_differences = np.square(train[left_indices] - train[right_indices])

    # Square the differences BEFORE taking the linearly interpolated quantile.
    # The same pairs and thresholds are used for all 16 ensemble/model pools.
    thresholds = {
        p: float(np.quantile(squared_differences, p / 100, method='linear'))
        for p in CLIPPING_PERCENTILES
    }
    assert all(np.isfinite(q) and q > 0 for q in thresholds.values())
    calibration = {
        'training_sample': '1990Q1--2007Q4', 'training_n': len(train),
        'evaluation_sample': '2008Q1--2019Q4', 'evaluation_n': len(targets),
        'pair_resamples': PAIR_RESAMPLES, 'pair_seed': PAIR_SEED,
        'quantile_method': 'linear',
        'definition': 'Quantile of squared differences between independent resampled training observations.',
        'squared_thresholds': {str(p): q for p, q in thresholds.items()},
        'raw_eta': ETA, 'raw_c0': C0,
        'scaled_eta': {str(p): ETA * q for p, q in thresholds.items()},
        'scaled_c0': {str(p): C0 * q for p, q in thresholds.items()},
        'reference_msfe': reference_msfe,
    }
    results_clipped = {
        'calibration': calibration,
        'calibration_draws': {
            'training_gdp': training,
            'left_indices': left_indices, 'right_indices': right_indices,
            'squared_differences': squared_differences,
        },
        'observed_gdp': targets,
        'pools': {},
    }

    #endregion

    #region --- Construct Losses and Recompute Forecast Combinations

    table_rows = {p: [] for p in CLIPPING_PERCENTILES}
    for ensemble, suffix in ENSEMBLES:
        results_clipped['pools'][ensemble] = {}
        for model, filename in MODELS:
            saved = load_results(
                PATH_RESULTS / f'results_M__multistep_{filename}_fix_grid_{suffix}.pkl'
            )
            assert len(saved[0]) == len(saved[1]) == 1
            forecasts = saved[0][0]
            assert forecasts.shape == (48, ESN_MATRIX_RESAMPLES)
            assert forecasts.index.equals(targets.index)

            # Raw expert losses: one squared error per quarter and expert.
            raw_losses = evaluation_losses(forecasts, targets)
            np.testing.assert_array_equal(raw_losses.to_numpy(), saved[1][0].to_numpy())
            raw_combined = combine_forecasts(forecasts, raw_losses, ETA, C0)
            raw_evaluation = evaluation_losses(raw_combined, targets)
            raw_msfe = raw_evaluation.mean(axis=0)
            pool = {
                'source_file': f'results_rev1/results_M__multistep_{filename}_fix_grid_{suffix}.pkl',
                'combination_losses': {'Raw/Raw': raw_losses},
                'combined_forecasts': {'Raw/Raw': raw_combined},
                'evaluation_losses': {'Raw/Raw': raw_evaluation},
            }

            for p in CLIPPING_PERCENTILES:
                q_p = thresholds[p]
                treatment = f'{p}/Raw'

                # Clipping and rescaling affect the online combination loss only.
                clipped_losses = np.minimum(raw_losses / q_p, 1.0)
                eta_p = ETA * q_p
                c0_p = C0 * q_p
                combined = combine_forecasts(forecasts, clipped_losses, eta_p, c0_p)
                squared_errors = evaluation_losses(combined, targets)
                msfe = squared_errors.mean(axis=0)

                pool['combination_losses'][treatment] = clipped_losses
                pool['combined_forecasts'][treatment] = combined
                pool['evaluation_losses'][treatment] = squared_errors

                # Relative MSFE uses the in-sample mean as in Table 2.
                # Percentage changes instead compare the same method with Raw/Raw.
                for method in METHODS:
                    table_rows[p].append({
                        'Ensemble': ensemble, 'Model': model, 'Method': method,
                        'Loss': treatment, 'MSFE': float(msfe[method]),
                        'Relative_MSFE': float(msfe[method] / reference_msfe),
                        'Raw_MSFE': float(raw_msfe[method]),
                        'Raw_Relative_MSFE': float(raw_msfe[method] / reference_msfe),
                        'Change_vs_raw_pct': float(100 * (msfe[method] / raw_msfe[method] - 1)),
                    })
            results_clipped['pools'][ensemble][model] = pool

    #endregion

    #region --- Save Losses and Construct Tables

    if SAVE_RESULTS:
        PATH_CLIPPED.mkdir(parents=True, exist_ok=True)
        with (PATH_CLIPPED / 'calibration.json').open('w', encoding='utf-8') as _file:
            json.dump(calibration, _file, indent=2, allow_nan=False)
        with (PATH_CLIPPED / 'results_M__clipped_losses.pkl').open('wb') as _file:
            pickle.dump(results_clipped, _file, protocol=pickle.HIGHEST_PROTOCOL)
        print('[!] Calibration, combination losses and forecast results saved to results_clipped_rev1/.')

    for number, p in enumerate(CLIPPING_PERCENTILES, start=1):
        table_data = pd.DataFrame(table_rows[p])
        if SAVE_CSVS:
            PATH_TABLES.mkdir(parents=True, exist_ok=True)
            path = PATH_TABLES / f'TableI{number}_medium_MD_clipped{p}.csv'
            table_data.to_csv(path, index=False)
            print(f'[!] Table saved as CSV:\n\t{path}')

        # Same console-table style as the main empirical scripts.
        _tmp_table = Table(title=f'Table I.{number}: Medium-MD -- {p}/Raw', box=box.MINIMAL)
        _tmp_table.add_column('Ensemble', no_wrap=True)
        _tmp_table.add_column('Model', no_wrap=True)
        for method in METHODS:
            _tmp_table.add_column(method, justify='right', no_wrap=True)
        for ensemble, _ in ENSEMBLES:
            for model, _ in MODELS:
                selected = table_data[(table_data.Ensemble == ensemble) & (table_data.Model == model)]
                cells = []
                for row in selected.itertuples(index=False):
                    change = '0.00' if row.Change_vs_raw_pct == 0 else f'{row.Change_vs_raw_pct:+.2f}'
                    cells.append(f'{row.Relative_MSFE:.3f} ({change}%)')
                _tmp_table.add_row(ensemble, model, *cells)
        console.print(_tmp_table)

    #endregion


if __name__ == '__main__':
    main()
