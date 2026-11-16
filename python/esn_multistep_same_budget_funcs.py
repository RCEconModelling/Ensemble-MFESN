"""
MFESN Same-Budget Numerical Functions

Both arms use their own reservoir states. Exact-input state/alignment
results are reused only within a draw and leak rate, as in the timed study.
"""

import copy
import hashlib
import inspect
import pickle
import time
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult
import newToolbox_ESN_Multi as mf_module
from esn_multistep_resampling_funcs import make_model
from esn_multistep_ensemble_funcs import esnSingle_fix_fit_multistep, esnMulti_fix_fit_multistep
from ensemble_tables_b1 import combination_forecasts
from ensemble_tables_b100 import METHODS

ALPHAS = np.array([.1, .3, .5, .7, .9])
LOG_LAMBDAS = np.array([-2., -1., 0., 1., 2.])

# region --- Within-Alpha State Reuse


def memoize_states(model):
    # Fresh caches for every alpha, draw and arm. Copies keep the original
    # helper from modifying a stored state. Keys include all exact inputs.
    for name in ['multifreq_states', 'multifreq_states_to_matrix']:
        original = getattr(model, name)
        signature = inspect.signature(original)
        cache = {}

        def cached(*args, _original=original, _signature=signature, _cache=cache, **kwargs):
            bound = _signature.bind(*args, **kwargs)
            bound.apply_defaults()
            key = hashlib.sha256(pickle.dumps(dict(bound.arguments), protocol=5)).digest()
            if key not in _cache:
                _cache[key] = copy.deepcopy(_original(*args, **kwargs))
            return copy.deepcopy(_cache[key])
        setattr(model, name, cached)


def at_alpha(base, alpha):
    model = copy.deepcopy(base)
    for reservoir in model.models_:
        reservoir.leak_rate_ = np.full_like(reservoir.leak_rate_, float(alpha))
    memoize_states(model)
    return model

# endregion

# region --- Original CV and Final Forecast Helpers


def fixed_cv(model, extended, single, logs):
    scores = []
    original = mf_module.scipy_minimize

    def evaluate(*args, **kwargs):
        objective = kwargs.get('fun', args[0] if args else None)
        if scores or not callable(objective):
            raise ValueError('Expected one original isotropic CV objective.')
        scores.extend(float(objective(np.array([float(x)]))) for x in logs)
        # Finish the original CV helper without optimizing or selecting lambda.
        return OptimizeResult(x=np.array([float(logs[0])]), fun=scores[0], success=True)
    inputs = (extended['md_fill_agg_data_train'],) if single else (
        extended['m_data_train'], extended['d_data_train'])
    mf_module.scipy_minimize = evaluate
    try:
        model.ridge_lambda_cv(Y=extended['GDP_fill_data_train'], z=inputs,
                              method='ridge-isotropic', cv_options='-cv:10-test_size:5', steps=1, debug=False)
    finally:
        mf_module.scipy_minimize = original
    if len(scores) != len(logs) or not np.isfinite(scores).all():
        raise ValueError('Invalid CV scores.')
    return np.asarray(scores)


def final_forecast(model, lam, dataset, single):
    if single:
        forecast = esnSingle_fix_fit_multistep(esnModel=model, Lambda=[np.array([lam])],
                                               data=dataset, steps=1, direct=False, aggregate=True)
    else:
        forecast = esnMulti_fix_fit_multistep(esnModel=model, Lambda=[lam, lam],
                                              data=dataset, steps=1, direct=False)
    return forecast.iloc[:, 0]


def elapsed(start):
    return dict(wall_seconds=time.perf_counter()-start[0], cpu_seconds=time.process_time()-start[1])


def total_timing(parts):
    return dict(wall_seconds=sum(x['wall_seconds'] for x in parts),
                cpu_seconds=sum(x['cpu_seconds'] for x in parts), components=parts)

# endregion

# region --- Two Numerical Arms


def tune_first_draw(model_name, seeds, dataset, extended):
    # The published grid is frozen; never resize it using forecast performance.
    alphas = np.linspace(.1, .9, 41)
    logs = np.linspace(-2., 2., 41)
    alphas[::10] = ALPHAS
    logs[::10] = LOG_LAMBDAS
    single = model_name.startswith('S-')
    start = time.perf_counter(), time.process_time()
    base = make_model(model_name, seeds, .1)
    parts = [elapsed(start)]
    rows = []
    for alpha in alphas:
        start = time.perf_counter(), time.process_time()
        model = at_alpha(base, float(alpha))
        rows.append(fixed_cv(model, extended, single, logs))
        parts.append(elapsed(start))
    start = time.perf_counter(), time.process_time()
    scores = np.asarray(rows)
    ai, li = np.unravel_index(np.argmin(scores), scores.shape)
    selected = at_alpha(base, float(alphas[ai]))
    forecast = final_forecast(selected, float(10.**logs[li]), dataset, single)
    parts.append(elapsed(start))
    arrays = dict(alpha_grid=alphas, log10_lambda_grid=logs, cv_objective_sum=scores,
                  selected_forecast=forecast.to_numpy(), selected_index=np.array([ai, li]),
                  first_seeds=np.asarray(seeds, dtype=np.uint64))
    if arrays['selected_forecast'].shape != (48,) or not np.isfinite(arrays['selected_forecast']).all():
        raise ValueError('Invalid selected forecast.')
    return arrays, total_timing(parts)


def build_ensemble(model_name, seeds, dataset):
    # 40 original draws x 5 alphas x 5 penalties. No benchmark-only CV here.
    single = model_name.startswith('S-')
    columns = []
    parts = []
    for pair in seeds:
        start = time.perf_counter(), time.process_time()
        base = make_model(model_name, pair, .1)
        draw = []
        for alpha in ALPHAS:
            model = at_alpha(base, float(alpha))
            for loglam in LOG_LAMBDAS:
                draw.append(final_forecast(model, float(10.**loglam), dataset, single).to_numpy())
        values = np.column_stack(draw)
        parts.append(elapsed(start))
        columns.append(values)
    start = time.perf_counter(), time.process_time()
    forecasts = pd.DataFrame(np.column_stack(columns), index=dataset['GDP_data_test'].index)
    y = dataset['GDP_data_test']
    losses = pd.DataFrame((forecasts.to_numpy()-y.to_numpy())**2, index=forecasts.index)
    combined = combination_forecasts(forecasts, losses, y)
    paths = np.column_stack([combined[name].to_numpy().ravel() for name in METHODS])
    parts.append(elapsed(start))
    return dict(expert_forecasts=forecasts.to_numpy(), combined_forecasts=paths), total_timing(parts)

# endregion
