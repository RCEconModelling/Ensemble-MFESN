"""
Forecast Combination with Random-Subset Experts

Default: reproduce Tables J.2 and J.3 from compact saved losses.
--simulate: generate data, fit experts and recompute all combinations.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'

import numpy as np
import pandas as pd
from simulation_random_subset_funcs import (
    METHODS, DESIGNS, SCENARIOS, GRID, B, BASE_SEED,
    generate_data, draw_subsets, fit_experts, evaluate_pool,
)
from simulation_tables_random_subset import make_tables, verify_manifest, check_case

#region // GLOBAL SETTINGS ------------------------------------

PATH_WD = Path(__file__).resolve().parents[1]
PATH_RESULTS = PATH_WD / 'results_random_subset_rev1'

#endregion

#region --- Run Simulations


def simulate_case(grid, replications, reference):
    regime, p, n = GRID[grid-1]
    scores = np.empty((2, 2, replications, 7))
    best = np.empty((2, 2, replications))
    for di, design in enumerate(DESIGNS):
        for b in range(replications):
            X, y = generate_data(design, n, b)
            subsets = draw_subsets(regime, p, b)
            forecasts = fit_experts(X, y, subsets, n)
            scores[di, :, b], best[di, :, b] = evaluate_pool(forecasts, y[n:])
            # Stop on numerical disagreement; do not replace failures by retries.
            np.testing.assert_allclose(scores[di, :, b], reference['method_msfe'][di, :, b],
                                       rtol=5e-11, atol=1e-9)
            np.testing.assert_allclose(best[di, :, b], reference['best_msfe'][di, :, b],
                                       rtol=5e-11, atol=1e-9)
            if (b+1) % 25 == 0 or b+1 == replications:
                print(f'Grid {grid}/27, {design}: {b+1}/{replications}', flush=True)
    return dict(method_msfe=scores, best_msfe=best, methods=np.array(METHODS),
                designs=np.array(DESIGNS), scenarios=np.array(SCENARIOS),
                regime=np.array(regime), p=np.array(p), T_train=np.array(n), T_test=np.array(1000),
                dgp_seeds=BASE_SEED + np.arange(replications),
                expert_seeds=BASE_SEED + 1000000 + np.arange(replications))


def run(args):
    if not 1 <= args.replications <= B:
        raise ValueError('Replications must be between 1 and 500.')
    destination = args.output.resolve()
    protected = PATH_RESULTS.resolve()
    if destination == PATH_WD.resolve() or destination == protected or protected in destination.parents:
        raise ValueError('Use a separate new output directory.')
    verify_manifest(PATH_RESULTS)
    destination.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(__file__).with_name('simulation_random_subset_funcs.py'),
               Path(__file__).with_name('simulation_tables_random_subset.py'),
               Path(__file__).with_name('comb_multistep_funcs.py'), PATH_RESULTS / 'manifest.json']
    config = dict(B=args.replications, numpy=np.__version__, pandas=pd.__version__,
                  hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    config_path = destination / 'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Configuration changed; use a new output folder.')
    lock = destination / 'run.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
        for grid in ([args.grid] if args.grid else range(1, 28)):
            target = destination / f'g{grid:02d}.npz'
            with np.load(PATH_RESULTS / target.name, allow_pickle=False) as reference:
                if target.exists():
                    with np.load(target, allow_pickle=False) as saved:
                        check_case(saved, grid, args.replications)
                        for name in ['method_msfe', 'best_msfe']:
                            np.testing.assert_allclose(saved[name], reference[name][:, :, :args.replications],
                                                       rtol=5e-11, atol=1e-9)
                    continue
                arrays = simulate_case(grid, args.replications, reference)
                check_case(arrays, grid, args.replications)
                temporary = destination / f'g{grid:02d}.tmp.npz'
                np.savez_compressed(temporary, **arrays)
                temporary.replace(target)
        if args.replications == B and all((destination / f'g{g:02d}.npz').exists() for g in range(1, 28)):
            files = {f'g{g:02d}.npz': hashlib.sha256((destination / f'g{g:02d}.npz').read_bytes()).hexdigest()
                     for g in range(1, 28)}
            (destination / 'manifest.json').write_text(
                json.dumps(dict(B=B, configuration=config, files=files), indent=2), encoding='utf-8')
            make_tables(destination, destination / 'tables')
    finally:
        lock.unlink()

#endregion

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--grid', type=int, choices=range(1, 28))
    parser.add_argument('--replications', type=int, default=B)
    parser.add_argument('--output', type=Path, default=PATH_WD / 'results_random_subset_regenerated')
    args = parser.parse_args()
    if args.simulate:
        run(args)
    else:
        make_tables(PATH_RESULTS, PATH_WD / 'tables_rev1')
