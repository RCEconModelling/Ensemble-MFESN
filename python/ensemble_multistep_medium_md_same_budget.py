"""
Ensemble MFESN Medium MD: Approximately Matched Computational Budgets

Default: reproduce Table I.7 from saved results.
--forecast: refit the 400 first-draw single models at the frozen 41x41 grid.
--check-timing: fresh paired checks on original pools 1, 2, 3 (zero-based).
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

# Same numerical resources for both arms; no process-level parallelism.
for variable in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[variable] = '1'

PATH_WD = Path(__file__).resolve().parents[1]
PATH_RESULTS = PATH_WD / 'results_same_budget_rev1'
PATH_B100 = PATH_WD / 'results_b100_rev1'


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def power_snapshot():
    # Observe power state without changing system settings.
    if os.name != 'nt':
        return dict(available=False)
    import ctypes

    class Power(ctypes.Structure):
        _fields_ = [('ac', ctypes.c_ubyte), ('flags', ctypes.c_ubyte),
                    ('percent', ctypes.c_ubyte), ('saving', ctypes.c_ubyte),
                    ('remaining', ctypes.c_uint32), ('full', ctypes.c_uint32)]
    value = Power()
    available = bool(ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(value)))
    return dict(available=available, ac_line_status=int(value.ac),
                power_saving=int(value.saving), recorded_epoch=time.time())


def check_selected(arrays, reference, pool):
    import numpy as np
    # Numerical mismatches stop the run; they are not timing-only retries.
    for name in ['alpha_grid', 'log10_lambda_grid']:
        np.testing.assert_array_equal(arrays[name], reference[name])
    for name, saved_name in [('cv_objective_sum', 'cv_objective_sum'),
                             ('selected_forecast', 'selected_forecasts'),
                             ('selected_index', 'selected_indices'),
                             ('first_seeds', 'first_seeds')]:
        np.testing.assert_array_equal(arrays[name], reference[saved_name][pool])


def forecast_case(model, tag, seeds, dataset, extended, destination, reference):
    import numpy as np
    from esn_multistep_same_budget_funcs import tune_first_draw
    folder = destination / tag
    folder.mkdir(exist_ok=True)
    pools = []
    for b in range(100):
        path = folder / f'pool_{b:03d}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                arrays = {name: saved[name].copy() for name in saved.files}
        else:
            print(f'{model}: fine-grid pool {b+1}/100', flush=True)
            arrays, timing = tune_first_draw(model, seeds[b, 0], dataset, extended)
            check_selected(arrays, reference, b)
            temporary = folder / f'pool_{b:03d}.tmp.npz'
            np.savez_compressed(temporary, **arrays)
            temporary.replace(path)
            save_json(folder / f'pool_{b:03d}_timing.json', timing)
        check_selected(arrays, reference, b)
        pools.append(arrays)
    y = dataset['GDP_data_test'].to_numpy().ravel()
    mean = float(dataset['GDP_data_train'].to_numpy().mean())
    np.savez_compressed(destination / f'{tag}.npz',
                        cv_objective_sum=np.asarray([p['cv_objective_sum'] for p in pools]),
                        selected_indices=np.asarray([p['selected_index'] for p in pools]),
                        selected_forecasts=np.asarray([p['selected_forecast'] for p in pools]),
                        first_seeds=seeds[:, 0], alpha_grid=pools[0]['alpha_grid'],
                        log10_lambda_grid=pools[0]['log10_lambda_grid'], observed=y,
                        in_sample_mean_forecast=np.full(48, mean), pool_ids=np.arange(100),
                        quarters=reference['quarters'])
    save_json(folder / 'status.json', dict(numerical_reproduction='passed', pools=100,
                                           grid=[41, 41], new_production_budget_validation=False,
                                           note='Refitting at the published grid does not establish a new matched-budget production experiment.'))


def timing_case(model, tag, seeds, dataset, extended, destination, reference, ensemble_reference):
    import numpy as np
    from statistics import median
    from esn_multistep_same_budget_funcs import (make_model, at_alpha, fixed_cv,
                                                 final_forecast, tune_first_draw, build_ensemble, LOG_LAMBDAS)
    from ensemble_same_budget_timing import timing_valid, repeat_spread

    # Warm up and check cached versus original helpers outside numerical timing.
    base = make_model(model, seeds[1, 0], .1)
    for alpha in [.1, .9]:
        cached = at_alpha(base, alpha)
        plain = make_model(model, seeds[1, 0], alpha)
        np.testing.assert_array_equal(fixed_cv(cached, extended, model.startswith('S-'), LOG_LAMBDAS),
                                      fixed_cv(plain, extended, model.startswith('S-'), LOG_LAMBDAS))
        np.testing.assert_array_equal(final_forecast(cached, .01, dataset, model.startswith('S-')),
                                      final_forecast(plain, .01, dataset, model.startswith('S-')))
    folder = destination / f'{tag}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
    folder.mkdir()
    before = power_snapshot()
    pools = []
    for pool in [1, 2, 3]:
        measured = {'ensemble': [], 'tuning': []}
        for repeat, order in [(1, ['ensemble', 'tuning']), (2, ['tuning', 'ensemble'])]:
            for arm in order:
                print(f'{model}: paired pool {pool}, repeat {repeat}, {arm}', flush=True)
                if arm == 'ensemble':
                    arrays, timing = build_ensemble(model, seeds[pool], dataset)
                else:
                    arrays, timing = tune_first_draw(model, seeds[pool, 0], dataset, extended)
                # Save every measurement before checking it; never discard failures.
                name = f'p{pool}_{arm}_{repeat}'
                save_json(folder / f'{name}_timing.json', timing)
                try:
                    if arm == 'tuning':
                        np.savez_compressed(folder / f'{name}.npz', **arrays)
                        check_selected(arrays, reference, pool)
                    else:
                        loss = np.mean((arrays['expert_forecasts'] -
                                       reference['observed'][:, None])**2, axis=0)
                        np.savez_compressed(folder / f'{name}.npz', expert_msfe=loss,
                                            combined_forecasts=arrays['combined_forecasts'],
                                            full_forecast_sha256=np.array(hashlib.sha256(arrays['expert_forecasts'].tobytes()).hexdigest()))
                        np.testing.assert_allclose(
                            loss, ensemble_reference['expert_msfe'][pool], rtol=1e-11, atol=1e-15)
                        np.testing.assert_allclose(
                            arrays['combined_forecasts'], ensemble_reference['combined_forecasts'][pool], rtol=1e-11, atol=1e-14)
                except Exception as error:
                    save_json(folder / 'failure.json',
                              dict(kind='numerical_identity', message=str(error)))
                    raise
                measured[arm].append(timing)
        e, u = measured['ensemble'], measured['tuning']
        ratio = median(x['wall_seconds'] for x in u)/median(x['wall_seconds'] for x in e)
        gross = all(timing_valid(x) and all(timing_valid(p) for p in x['components']) for x in e+u)
        pools.append(dict(pool=pool, ratio=ratio, ensemble_spread=repeat_spread(e),
                          tuning_spread=repeat_spread(u), wall_cpu_checks=gross,
                          passed=.9 <= ratio <= 1.1 and repeat_spread(e) <= .2 and repeat_spread(u) <= .2 and gross))
        save_json(folder / 'progress.json', dict(pools=pools))
    after = power_snapshot()
    power_ok = before['available'] and after['available'] and all(
        before[k] == after[k] for k in ['ac_line_status', 'power_saving'])
    passed = power_ok and all(p['passed'] for p in pools)
    save_json(folder / 'gate.json', dict(status='passed' if passed else 'not_passed',
                                         pools=pools, grid=[41, 41], fresh_both_arms=True, power_before=before, power_after=after,
                                         scope='Three representative paired pools; not a new 400-pair production run.'))
    if not passed:
        raise RuntimeError(
            f'Timing check did not pass; retained at {folder}. Grids and tolerances were not changed.')
    print(f'Fresh representative timing gate passed: {folder}')


def run(args):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import numpy as np
    import pandas as pd
    import scipy
    from ensemble_tables_b100 import MODELS
    from ensemble_tables_same_budget import make_table, verify_manifest
    from esn_multistep_resampling_funcs import medium_datasets
    verify_manifest(PATH_RESULTS, PATH_B100)
    destination = args.output.resolve()
    protected = [PATH_RESULTS.resolve(), PATH_B100.resolve()]
    if destination == PATH_WD.resolve() or any(
            destination == folder or folder in destination.parents for folder in protected):
        raise ValueError('Use a separate new output directory.')
    destination.mkdir(parents=True, exist_ok=True)
    sources = list(Path(__file__).parent.glob('*.py')) + [args.data/'fullsample_output_data.csv',
                                                          PATH_RESULTS/'manifest.json', PATH_B100/'reservoir_seeds.npz']
    config = dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                  hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, grid=[41, 41])
    config_path = destination / 'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Data/code configuration changed; use a new output folder.')
    # Exclusive local guard; a stale guard requires inspection, not automatic removal.
    lock = destination / 'run.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        save_json(config_path, config)
        dataset, extended = medium_datasets(args.data)
        with np.load(PATH_B100/'reservoir_seeds.npz', allow_pickle=False) as schedule:
            for model, tag in MODELS:
                if args.model and tag != args.model:
                    continue
                with np.load(PATH_RESULTS/f'{tag}.npz', allow_pickle=False) as reference, np.load(
                        PATH_B100/f'{tag}_alpha-lambdaRP.npz', allow_pickle=False) as ensemble:
                    if args.check_timing:
                        timing_case(model, tag, schedule[tag+'_alpha-lambdaRP'],
                                    dataset, extended, destination, reference, ensemble)
                    else:
                        forecast_case(
                            model, tag, schedule[tag+'_alpha-lambdaRP'], dataset, extended, destination, reference)
        if args.forecast and all((destination/f'{tag}.npz').exists() for _, tag in MODELS):
            make_table(destination, PATH_B100, destination/'tables', verify_hashes=False)
    finally:
        lock.unlink()  # Only this invocation's temporary guard.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--forecast', action='store_true')
    mode.add_argument('--check-timing', action='store_true')
    parser.add_argument('--model', choices=['smfesn_a', 'smfesn_b', 'mmfesn_a', 'mmfesn_b'])
    parser.add_argument('--data', type=Path, default=PATH_WD/'data/preprocessed')
    parser.add_argument('--output', type=Path, default=PATH_WD/'results_same_budget_regenerated')
    args = parser.parse_args()
    if args.forecast or args.check_timing:
        run(args)
    else:
        from ensemble_tables_same_budget import make_table
        make_table(PATH_RESULTS, PATH_B100, PATH_WD/'tables_rev1')
