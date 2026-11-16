"""
Forecast Combination with Simulated AR(1) Losses

Default: reproduce Figures J.1 and J.2 from compact saved regrets.
--simulate: regenerate the losses and recompute FTL and decreasing Hedge.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'

import numpy as np
from simulation_ar1_funcs import (
    B, K, T, Q_GRID, DELTA_GRID, MU_STAR, SEPARATION, A, C0, MASTER_SEED,
    STATIONARY_TERMS, simulate_losses, verify_results, summarize_results,
)
from simulation_ar1_plots import make_figures

#region // GLOBAL SETTINGS ------------------------------------

PATH_WD = Path(__file__).resolve().parents[1]
PATH_RESULTS = PATH_WD / 'results_ar1_rev1'

#endregion

#region --- Results and Simulations


def export_results(source, tables, figures):
    summary = summarize_results(source)
    tables.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables / 'AR1_loss_regret_summary.csv', index=False, float_format='%.17g')
    make_figures(summary, figures)


def run(args):
    if not 1 <= args.replications <= B:
        raise ValueError('Replications must be between 1 and 500.')
    # A new directory preserves the distributed results and previous reruns.
    destination = args.output.resolve()
    if PATH_RESULTS.resolve() in destination.parents:
        raise ValueError('Use a separate output directory, outside the saved results.')
    destination.mkdir(parents=True, exist_ok=False)
    config = dict(B=args.replications, K=K, T=T, mu_star=MU_STAR, separation=SEPARATION,
                  scale=A, c0=C0, master_seed=MASTER_SEED, stationary_terms=STATIONARY_TERMS,
                  q_grid=Q_GRID if args.q is None else [args.q], delta_grid=DELTA_GRID.tolist(),
                  numpy=np.__version__)
    (destination / 'configuration.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    files = {}
    for q in config['q_grid']:
        name = f'q_{q:.2f}.npz'
        results = simulate_losses(q, args.replications)
        # Check against the retained manuscript run before saving a completed case.
        with np.load(PATH_RESULTS / name, allow_pickle=False) as reference:
            np.testing.assert_allclose(results['regret'], reference['regret'][:args.replications],
                                       rtol=1e-11, atol=1e-10)
        np.savez_compressed(destination / name, **results)
        files[name] = hashlib.sha256((destination / name).read_bytes()).hexdigest()
    (destination / 'manifest.json').write_text(
        json.dumps(dict(configuration=config, files=files), indent=2), encoding='utf-8')
    if args.replications == B and args.q is None:
        export_results(destination, destination / 'tables', destination / 'figures')

#endregion

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--q', type=float, choices=Q_GRID)
    parser.add_argument('--replications', type=int, default=B)
    parser.add_argument('--output', type=Path, default=PATH_WD / 'results_ar1_regenerated')
    args = parser.parse_args()
    verify_results(PATH_RESULTS)
    if args.simulate:
        run(args)
    else:
        if args.q is not None or args.replications != B or args.output != PATH_WD / 'results_ar1_regenerated':
            parser.error('--q, --replications and --output require --simulate.')
        export_results(PATH_RESULTS, PATH_WD / 'tables_rev1', PATH_WD / 'figures_rev1/ar1')
