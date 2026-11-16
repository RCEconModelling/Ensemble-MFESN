"""
Ensemble MFESN Medium MD: Robustness to Expert Resampling

B=100, K=1000, h=1. By default, reproduce Tables I.5 and I.6 from
compact saved results. Use --forecast explicitly to generate the pools.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import warnings

# One numerical worker, one BLAS thread.
for variable in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[variable] = '1'
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from ensemble_tables_b1 import combination_forecasts
from ensemble_tables_b100 import MODELS, ENSEMBLES, METHODS, make_tables
from esn_multistep_resampling_funcs import medium_datasets, fit_pool

#region // GLOBAL SETTINGS ------------------------------------

PATH_WD = Path(__file__).resolve().parents[1]
PATH_RESULTS = PATH_WD / 'results_b100_rev1'
B = 100
K = 1000

#endregion

#region --- Repeated Expert Pools

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate_results(path_data, destination, model_filter=None, ensemble_filter=None):
    destination = Path(destination)
    if destination.resolve() == PATH_RESULTS.resolve():
        raise ValueError('Use a separate output folder; retain the supplied results.')
    destination.mkdir(parents=True, exist_ok=True)
    # Refuse to mix checkpoints from different data, seeds, code or versions.
    import scipy
    sources = [Path(__file__), Path(__file__).with_name('esn_multistep_resampling_funcs.py'),
               Path(__file__).with_name('ensemble_tables_b1.py'),
               Path(__file__).with_name('ensemble_tables_b100.py')]
    sources += [Path(__file__).with_name(name) for name in ['newToolbox_ESN.py',
                'newToolbox_ESN_Multi.py', 'data_funcs.py', 'esn_multistep_ensemble_funcs.py',
                'comb_multistep_funcs.py']]
    sources += [Path(path_data)/'fullsample_output_data.csv', PATH_RESULTS/'reservoir_seeds.npz']
    configuration = dict(B=B, K=K, h=1, numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                         source_hashes={p.name: sha256(p) for p in sources})
    config_path = destination / 'configuration.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != configuration:
            raise ValueError('Checkpoint configuration changed; use a new output folder.')
    else:
        config_path.write_text(json.dumps(configuration, indent=2), encoding='utf-8')
    dataset, extended = medium_datasets(path_data)
    targets = dataset['GDP_data_test']
    y = targets.to_numpy().ravel()
    benchmark = float(np.mean((y-float(dataset['GDP_data_train'].to_numpy().mean()))**2))
    with np.load(PATH_RESULTS / 'reservoir_seeds.npz', allow_pickle=False) as schedule:
        for model, tag in MODELS:
            if model_filter and tag != model_filter:
                continue
            for ensemble in ENSEMBLES:
                if ensemble_filter and ensemble != ensemble_filter:
                    continue
                key = f'{tag}_{ensemble}'
                seeds = schedule[key]
                folder = destination / 'pools' / key
                folder.mkdir(parents=True, exist_ok=True)
                pools = []
                for b in range(B):
                    path = folder / f'pool_{b:03d}.npz'
                    if path.exists():
                        with np.load(path, allow_pickle=False) as saved:
                            pool = {name: saved[name].copy() for name in saved.files}
                        np.testing.assert_array_equal(pool['seeds'], seeds[b])
                    else:
                        print(f'{model}, {ensemble}: pool {b+1}/{B}', flush=True)
                        forecasts, cv_scores, penalties, audit = fit_pool(model, ensemble, seeds[b], dataset, extended)
                        if forecasts.shape != (48, K):
                            raise ValueError('Expected 48 quarters and 1000 experts.')
                        losses = pd.DataFrame((forecasts.to_numpy()-y[:,None])**2, index=forecasts.index)
                        if not forecasts.index.to_period('Q').equals(targets.index.to_period('Q')):
                            raise ValueError('Forecast quarters differ from target quarters.')
                        aligned_targets = pd.DataFrame(y, index=forecasts.index)
                        combined = combination_forecasts(forecasts, losses, aligned_targets)
                        paths = np.column_stack([combined[name].to_numpy().ravel() for name in METHODS])
                        selected = int(np.argmin(cv_scores))
                        expert_msfe = losses.to_numpy().mean(axis=0)
                        pool = dict(cv_scores=cv_scores, ridge_penalties=penalties, expert_msfe=expert_msfe,
                            selected_ids=np.array(selected), selected_forecasts=forecasts.iloc[:,selected].to_numpy(),
                            combined_forecasts=paths, cv_msfe=np.array(expert_msfe[selected]),
                            method_msfe=np.mean((paths-y[:,None])**2, axis=0), seeds=seeds[b], observed=y)
                        (folder/f'pool_{b:03d}_cv.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
                        temporary = folder / f'pool_{b:03d}.tmp.npz'
                        np.savez_compressed(temporary, **pool)
                        temporary.replace(path)
                    np.testing.assert_array_equal(pool['observed'], y)
                    for name, shape in [('cv_scores',(K,)), ('ridge_penalties',(K,)),
                                        ('expert_msfe',(K,)), ('combined_forecasts',(48,7)),
                                        ('selected_forecasts',(48,)), ('method_msfe',(7,))]:
                        if pool[name].shape != shape or not np.isfinite(pool[name]).all():
                            raise ValueError(f'Invalid saved pool checkpoint: {name}')
                    selected = int(np.argmin(pool['cv_scores']))
                    if int(pool['selected_ids']) != selected:
                        raise ValueError('CV-selected expert differs from saved checkpoint.')
                    np.testing.assert_allclose(pool['cv_msfe'], pool['expert_msfe'][selected], rtol=1e-12)
                    np.testing.assert_allclose(pool['method_msfe'],
                        np.mean((pool['combined_forecasts']-y[:,None])**2, axis=0), rtol=1e-12)
                    pools.append(pool)
                arrays = {name: np.asarray([pool[name] for pool in pools]) for name in pools[0]
                          if name not in ['observed', 'seeds']}
                np.savez_compressed(destination / f'{key}.npz', **arrays, observed=y,
                    methods=np.asarray(METHODS), quarters=np.asarray(targets.index.to_period('Q').astype(str), dtype='U6'),
                    benchmark_msfe=np.array(benchmark))
    if all((destination/f'{tag}_{ensemble}.npz').exists() for _,tag in MODELS for ensemble in ENSEMBLES):
        make_tables(destination, destination/'tables')

#endregion

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forecast', action='store_true', help='Refit all requested pools; potentially a long run.')
    parser.add_argument('--data', type=Path, default=PATH_WD/'data/preprocessed')
    parser.add_argument('--output', type=Path, default=PATH_WD/'results_b100_regenerated')
    parser.add_argument('--model', choices=[tag for _,tag in MODELS])
    parser.add_argument('--ensemble', choices=ENSEMBLES)
    args = parser.parse_args()
    if args.forecast:
        generate_results(args.data, args.output, args.model, args.ensemble)
    else:
        make_tables(PATH_RESULTS, PATH_WD/'tables_rev1')
