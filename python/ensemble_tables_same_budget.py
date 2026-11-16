"""
Medium-MD Same-Budget Table (Table I.7)

Reuses corrected B=100 alpha-lambdaRP combination forecasts. The EB column
uses the finer-grid single model, not the pool's CV-selected expert.
"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ensemble_tables_b100 import MODELS, METHODS, check_results

PATH_WD = Path(__file__).resolve().parents[1]


def verify_manifest(path_results, path_b100):
    manifest = json.loads((Path(path_results)/'manifest.json').read_text())
    for root, files in [(path_results, manifest['files']), (path_b100, manifest['b100_dependencies'])]:
        for name, expected in files.items():
            if hashlib.sha256((Path(root)/name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Saved result hash differs: {name}')


def check_selected(selected, ensemble, seeds):
    alphas = np.linspace(.1, .9, 41)
    logs = np.linspace(-2., 2., 41)
    alphas[::10] = [.1, .3, .5, .7, .9]
    logs[::10] = [-2, -1, 0, 1, 2]
    np.testing.assert_array_equal(selected['alpha_grid'], alphas)
    np.testing.assert_array_equal(selected['log10_lambda_grid'], logs)
    np.testing.assert_array_equal(selected['first_seeds'], seeds[:, 0])
    np.testing.assert_array_equal(selected['observed'], ensemble['observed'])
    np.testing.assert_array_equal(selected['quarters'], ensemble['quarters'])
    np.testing.assert_array_equal(selected['pool_ids'], np.arange(100))
    for name, shape in [('cv_objective_sum', (100, 41, 41)), ('selected_forecasts', (100, 48)),
                        ('selected_indices', (100, 2))]:
        if selected[name].shape != shape or not np.isfinite(selected[name]).all():
            raise ValueError(f'Invalid saved {name}.')
    indices = np.asarray([np.unravel_index(np.argmin(s), (41, 41))
                         for s in selected['cv_objective_sum']])
    np.testing.assert_array_equal(indices, selected['selected_indices'])
    anchors = selected['cv_objective_sum'][:, ::10, ::10].reshape(100, 25)
    np.testing.assert_allclose(anchors/10, ensemble['cv_scores'][:, :25], rtol=1e-11, atol=1e-12)


def make_table(path_results, path_b100, path_tables, verify_hashes=True):
    path_results, path_b100, path_tables = map(Path, [path_results, path_b100, path_tables])
    if verify_hashes:
        verify_manifest(path_results, path_b100)
    rows = []
    paired = []
    with np.load(path_b100/'reservoir_seeds.npz', allow_pickle=False) as schedule:
        for model, tag in MODELS:
            with np.load(path_results/f'{tag}.npz', allow_pickle=False) as selected, np.load(
                    path_b100/f'{tag}_alpha-lambdaRP.npz', allow_pickle=False) as ensemble:
                check_results(ensemble)
                check_selected(selected, ensemble, schedule[tag+'_alpha-lambdaRP'])
                eb = np.mean((selected['selected_forecasts']-selected['observed'])**2, axis=1)
                benchmark = float(
                    np.mean((selected['in_sample_mean_forecast']-selected['observed'])**2))
                np.testing.assert_allclose(benchmark, ensemble['benchmark_msfe'], rtol=1e-12)
                means = ensemble['method_msfe'].mean(axis=0)
                order = np.argsort(means, kind='stable')
                row = dict(Model=model, EB=eb.mean()/benchmark)
                row.update(zip(METHODS, means/eb.mean()))
                row.update(Best=METHODS[order[0]], Second_best=METHODS[order[1]])
                rows.append(row)
                for b in range(100):
                    ai, li = selected['selected_indices'][b]
                    item = dict(Model=model, Replication=b+1, EB_MSFE=eb[b],
                                Selected_alpha=float(selected['alpha_grid'][ai]),
                                Selected_lambda=float(10.**selected['log10_lambda_grid'][li]))
                    for j, method in enumerate(METHODS):
                        item[method+'_MSFE'] = ensemble['method_msfe'][b, j]
                        item[method+'_minus_EB'] = ensemble['method_msfe'][b, j]-eb[b]
                    paired.append(item)
    path_tables.mkdir(parents=True, exist_ok=True)
    for filename, data in [('TableI7_medium_MD_same_budget.csv', rows),
                           ('medium_MD_same_budget_paired_losses.csv', paired)]:
        pd.DataFrame(data).to_csv(path_tables/filename, index=False, float_format='%.17g')
        print(f'Saved {path_tables/filename}')
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=PATH_WD/'results_same_budget_rev1')
    parser.add_argument('--b100', type=Path, default=PATH_WD/'results_b100_rev1')
    parser.add_argument('--tables', type=Path, default=PATH_WD/'tables_rev1')
    args = parser.parse_args()
    make_table(args.results, args.b100, args.tables)
