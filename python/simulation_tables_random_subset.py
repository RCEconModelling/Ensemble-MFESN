"""
Random-Subset Simulation Tables (Tables J.2 and J.3)

Ratios of mean MSFEs over designs and replications, not mean ratios.
Reads compact saved losses; no simulations are run by this script.
"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from simulation_random_subset_funcs import METHODS, DESIGNS, SCENARIOS, GRID, BASE_SEED, B

PATH_WD = Path(__file__).resolve().parents[1]

#region --- Check Saved Results


def verify_manifest(path):
    manifest = json.loads((Path(path) / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256((Path(path) / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Saved result hash differs: {name}')


def check_case(data, grid, replications=B):
    regime, p, n = GRID[grid-1]
    for name, expected in [('regime', regime), ('p', p), ('T_train', n), ('T_test', 1000)]:
        if data[name].item() != expected:
            raise ValueError(f'Unexpected {name} in grid {grid}.')
    for name, expected in [('methods', METHODS), ('designs', DESIGNS), ('scenarios', SCENARIOS)]:
        if data[name].tolist() != expected:
            raise ValueError(f'Unexpected {name} order.')
    for name, shape in [('method_msfe', (2, 2, replications, 7)),
                        ('best_msfe', (2, 2, replications))]:
        values = data[name]
        if values.shape != shape or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f'Invalid {name}.')
    np.testing.assert_array_equal(data['dgp_seeds'], BASE_SEED + np.arange(replications))
    np.testing.assert_array_equal(data['expert_seeds'], BASE_SEED + 1000000 + np.arange(replications))

#endregion

#region --- Construct Tables


def make_tables(path_results, path_tables, verify_hashes=True):
    path_results, path_tables = Path(path_results), Path(path_tables)
    if verify_hashes:
        verify_manifest(path_results)
    cases = []
    for grid, (regime, p, n) in enumerate(GRID, 1):
        with np.load(path_results / f'g{grid:02d}.npz', allow_pickle=False) as data:
            check_case(data, grid)
            cases.append((regime, n, data['method_msfe'].copy(), data['best_msfe'].copy()))
    path_tables.mkdir(parents=True, exist_ok=True)
    audit = []
    for si, scenario in enumerate(SCENARIOS):
        rows = []
        for letter, panel in zip('ABCD', ['pooled', 'small_only', 'mixed', 'large_only']):
            for di, design in enumerate(DESIGNS):
                for n in [100, 300, 500]:
                    selected = [c for c in cases if c[1] == n and (panel == 'pooled' or c[0] == panel)]
                    method = np.concatenate([c[2][di, si] for c in selected]).mean(axis=0)
                    best = np.concatenate([c[3][di, si] for c in selected]).mean()
                    relative = method / best
                    order = np.argsort(relative, kind='stable')
                    row = dict(Panel=letter, Subset_regime=panel, DGP=design, T_train=n)
                    row.update(zip(METHODS, relative))
                    row.update(Best=METHODS[order[0]], Second_best=METHODS[order[1]])
                    rows.append(row)
                    for mi, name in enumerate(METHODS):
                        audit.append(dict(Scenario=scenario, Panel=letter, DGP=design, T_train=n,
                            Method=name, Replications=len(selected)*B, Mean_MSFE=method[mi],
                            Mean_best_expert_MSFE=best, Relative_MSFE=relative[mi]))
        filename = f'TableJ{si+2}_random_subset_{scenario}.csv'
        pd.DataFrame(rows).to_csv(path_tables / filename, index=False, float_format='%.17g')
        print(f'Saved {path_tables / filename}')
    pd.DataFrame(audit).to_csv(path_tables / 'random_subset_mean_losses.csv',
                             index=False, float_format='%.17g')
    return pd.DataFrame(audit)

#endregion

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=PATH_WD / 'results_random_subset_rev1')
    parser.add_argument('--tables', type=Path, default=PATH_WD / 'tables_rev1')
    args = parser.parse_args()
    make_tables(args.results, args.tables)
