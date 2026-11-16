"""
ESN Ensemble Functions for Repeated Medium-MD Pools

Reuses the original data, CV, fitting and forecasting helpers.
Adds explicit reservoir seeds and saves the CV score of each expert.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult

import newToolbox_ESN_Multi as mf_module
from newToolbox_ESN import ESN, stateMatrixGenerator
from newToolbox_ESN_Multi import ESNMultiFrequency
from data_funcs import (daily_begginingInterp, daily_aggregateByIndex,
                        get_Medium_Datasets, get_Medium_Datasets_extended)
from esn_multistep_ensemble_funcs import esnSingle_fix_fit_multistep, esnMulti_fix_fit_multistep

#region --- Medium-MD Data

def medium_datasets(path_data):
    # Same preprocessing as the main Medium-MD script; keep the two GDP
    # calendars separate (business-quarter dates and calendar-quarter dates).
    preprocessed = pd.read_csv(Path(path_data) / 'fullsample_output_data.csv',
                               parse_dates=['Date']).set_index('Date')
    start, end = pd.Timestamp('1990-01-01'), pd.Timestamp('2019-12-31')
    data = preprocessed.loc[start:end]
    daily = data[[f'D{i}' for i in range(1, 14)]].dropna()
    monthly = data[[f'M{i}' for i in range(1, 21)]].dropna()
    gdp = data[['Q1']].dropna()
    daily_interp, daily_fill = pd.DataFrame(), pd.DataFrame()
    for column in daily.columns:
        series = daily_begginingInterp(daily[column], length=24, endOfMonth=False)
        daily_interp = daily_interp.join(series.rename(columns={0: column}), how='right')
        series = daily_begginingInterp(daily[column], length=24, endOfMonth=True)
        daily_fill = daily_fill.join(series.rename(columns={0: column}), how='right')
    daily_agg = daily_aggregateByIndex(daily_fill, window_size=6, endOfWindow=True)
    monthly_prefill = preprocessed[[f'M{i}' for i in range(1, 21)]].loc[
        (start - pd.offsets.MonthBegin()):end].dropna()
    monthly_prefill.index = monthly_prefill.index + pd.offsets.MonthEnd(0)
    pre_index = pd.date_range(start=start-pd.offsets.YearBegin(), end=end, freq='D')
    placeholder = pd.DataFrame(np.full((len(pre_index), 1), np.nan), index=pre_index)
    monthly_interp = placeholder.join(monthly_prefill).ffill().iloc[:,1:]
    gdp_fill = gdp.copy()
    gdp_fill.index = gdp_fill.index + pd.offsets.QuarterEnd(0)
    dataset = get_Medium_Datasets(
        data={'GDP_data': gdp, 'm_data': monthly, 'd_data_interp': daily_interp,
              'GDP_fill_data': gdp_fill, 'md_fill_data': daily_fill.join(monthly_interp),
              'md_fill_agg_data': daily_agg.join(monthly_interp)},
        startSampleDate=start, endSampleDate=end, trainTestSplitDate='2007-12-31')
    extended = get_Medium_Datasets_extended(preproc_data=preprocessed,
        startSampleDate='1975-01-01', endSampleDate='2019-12-31', trainTestSplitDate='2007-12-31')
    if len(dataset['GDP_data_test']) != 48:
        raise ValueError('Expected 48 quarterly forecasts.')
    return dataset, extended

#endregion

#region --- Model Construction

def make_model(model, seeds, alpha=None):
    # Within a draw, rebuild the same matrices when changing the leak rate.
    # For M-MFESNs the seed columns are monthly and daily, respectively.
    if model.startswith('S-'):
        settings = [(30 if model.endswith('A') else 120, 33, .5, 1., .1, seeds[0])]
    elif model == 'M-MFESN A':
        settings = [(100, 20, .5, 1.5, 0., seeds[0]), (20, 13, .5, .5, .1, seeds[1])]
    elif model == 'M-MFESN B':
        settings = [(100, 20, .08, .25, .3, seeds[0]), (20, 13, .01, .01, .99, seeds[1])]
    else:
        raise ValueError('Unknown model specification.')
    reservoirs = []
    for n, p, rho, gamma, default_alpha, seed in settings:
        reservoirs.append(ESN(
            N=n,
            A=stateMatrixGenerator((n,n), dist='sparse_normal', sparsity=10/n,
                                   normalize='eig', seed=int(seed)),
            C=stateMatrixGenerator((n,p), dist='sparse_uniform', sparsity=10/n,
                                   normalize='norm2', seed=int(seed)),
            rho=rho, gamma=gamma, leak_rate=default_alpha if alpha is None else alpha,
            activation=np.tanh))
    return ESNMultiFrequency(tuple(reservoirs), ar=False)

#endregion

#region --- Cross-Validation

def cv_penalties(model, extended, single, fixed_grid=None):
    """Use the original ten-fold objective, either optimizing or evaluating it."""
    original = mf_module.scipy_minimize
    scores, audit = [], []

    def record(*args, **kwargs):
        objective = kwargs.get('fun', args[0] if args else None)
        if scores:
            raise ValueError('Expected one isotropic CV objective.')
        if fixed_grid is not None:
            # Evaluate all assigned penalties. Do not optimize over lambda.
            scores.extend(float(objective(np.array([np.log10(lam)]))) for lam in fixed_grid)
            return OptimizeResult(x=np.array([np.log10(fixed_grid[0])]), fun=scores[0], success=True)

        trace = []
        def observe(x, *extra):
            value = float(objective(x, *extra))
            trace.append((np.array(x, copy=True), value))
            return value
        if 'fun' in kwargs:
            result = original(*args, **dict(kwargs, fun=observe))
        else:
            result = original(observe, *args[1:], **kwargs)
        replay = float(objective(result.x.copy()))
        np.testing.assert_allclose(replay, objective(result.x.copy()), rtol=1e-9, atol=1e-9)
        recorded = [value for x, value in trace if np.array_equal(x, result.x)]
        if not recorded:
            raise ValueError('Returned CV point was not evaluated.')
        np.testing.assert_allclose(recorded, replay, rtol=1e-9, atol=1e-9)
        discrepancy = not np.isclose(replay, result.fun, rtol=1e-9, atol=1e-9)
        if discrepancy:
            # Retain the verified SciPy 1.14 abnormal-exit reconciliation for M.
            # Never change the returned penalty or retry the optimization.
            known_exit = (not result.success and result.status == 2 and
                          str(result.message) == 'ABNORMAL_TERMINATION_IN_LNSRCH')
            matching_trial = any(not np.array_equal(x, result.x) and
                                 np.isclose(v, result.fun, rtol=1e-9, atol=1e-9) for x,v in trace)
            if single or not known_exit or not matching_trial:
                raise ValueError('Unexplained CV objective discrepancy.')
        # The retained S runs recorded optimizer.fun; the M runs scored the
        # returned penalty directly. Preserve both conventions and audit status.
        scores.append(float(result.fun) if single else replay)
        audit.append(dict(success=bool(result.success), status=int(result.status),
                          message=str(result.message), reported=float(result.fun),
                          replay=replay, reconciled=bool(discrepancy)))
        return result

    inputs = (extended['md_fill_agg_data_train'],) if single else (
        extended['m_data_train'], extended['d_data_train'])
    mf_module.scipy_minimize = record
    try:
        penalties = model.ridge_lambda_cv(Y=extended['GDP_fill_data_train'], z=inputs,
            method='ridge-isotropic', cv_options='-cv:10-test_size:5', steps=1, debug=False)
    finally:
        mf_module.scipy_minimize = original
    penalties = np.asarray(penalties).ravel() if fixed_grid is None else np.asarray(fixed_grid)
    if len(scores) != len(penalties) or not np.isfinite(scores).all() or np.any(np.asarray(scores) < 0):
        raise ValueError('Invalid CV scores.')
    return penalties, np.asarray(scores)/10., audit

#endregion

#region --- Fit and Forecast a Pool

def fit_pool(model_name, ensemble, seeds, dataset, extended):
    forecasts, scores, penalties, optimizer_audits = [], [], [], []
    single = model_name.startswith('S-')
    alphas = [.1, .3, .5, .7, .9] if ensemble in ['alphaRP', 'alpha-lambdaRP'] else [None]
    fixed_grid = np.logspace(-2, 2, 5) if ensemble in ['lambdaRP', 'alpha-lambdaRP'] else None
    for draw, pair in enumerate(seeds):
        for alpha in alphas:
            model = make_model(model_name, pair, alpha)
            lambdas, cv_scores, audit = cv_penalties(model, extended, single, fixed_grid)
            for lam, score in zip(lambdas, cv_scores):
                if single:
                    forecast = esnSingle_fix_fit_multistep(esnModel=model, Lambda=[np.array([lam])],
                        data=dataset, steps=1, direct=False, aggregate=True)
                else:
                    forecast = esnMulti_fix_fit_multistep(esnModel=model, Lambda=[lam, lam],
                        data=dataset, steps=1, direct=False)
                forecasts.append(forecast.iloc[:,0])
                scores.append(score)
                penalties.append(lam)
            optimizer_audits.append(dict(draw=draw, alpha=alpha, optimizer=audit))
        print(f'  Completed draw {draw+1}/{len(seeds)}', flush=True)
    forecasts = pd.concat(forecasts, axis=1)
    forecasts.columns = np.arange(forecasts.shape[1])
    if not np.isfinite(forecasts.to_numpy()).all():
        raise ValueError('Non-finite forecasts.')
    return forecasts, np.asarray(scores), np.asarray(penalties), optimizer_audits

#endregion
