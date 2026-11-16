# Reproducing the manuscript results

[Back to the repository overview](README.md)

The supplied saved results reproduce the tables and figures without the original
GDP predictor data or refitting. Dependencies are listed in
[pyproject.toml](pyproject.toml). Run commands from the repository root unless
specified otherwise.

GDP results use one-step-ahead forecasts over all 48 quarters, 2008Q1–2019Q4.
Use the `*_rev1/` results below, not the historical archives.

## 1. Tables from saved results

Each command produces all tables listed in its row.

| Tables | Command | Required saved results |
| --- | --- | --- |
| 2 and I.3: Medium-MD and Small-MD, B=1 | `python python/ensemble_tables_b1.py` | `results_rev1/`, `results_small_rev1/` |
| I.1 and I.2: clipped losses | `python python/ensemble_multistep_medium_md_clipped.py` | `results_rev1/` |
| I.5 and I.6: Medium-MD, B=100 | `python python/ensemble_tables_b100.py` | `results_b100_rev1/` |
| I.7: equal-budget comparison | `python python/ensemble_multistep_medium_md_same_budget.py` | `results_same_budget_rev1/`, `results_b100_rev1/` |
| J.2 and J.3: random-subset simulations | `python python/simulation_random_subset.py` | `results_random_subset_rev1/` |

CSVs are written to [tables_rev1/](tables_rev1/), replacing existing exports.
The clipped-loss command also recreates `results_clipped_rev1/`; use a separate
working copy to preserve the supplied files. Statistical definitions are in the
[exercise notes](#4-exercise-notes).

## 2. Figures from saved results

| Figures | Script |
| --- | --- |
| 1, 2, I.1–I.3, I.8–I.9: Medium-MD | `python/ensemble_multistep_medium_md.py` |
| I.4–I.7: Small-MD | `python/ensemble_multistep_small_md.py` |
| J.1–J.2: AR(1) losses | `python/simulation_ar1_losses.py` |

### GDP figures

In the relevant Medium-MD or Small-MD script, set these existing variables:

```python
FORECAST_STEPS = 1
OUTPUT_STEPS = [1]
ESN_MATRIX_RESAMPLES = 1000
DO_FORECASTS = False
DO_PLOTS = True
OPTIONAL_PLOTS = False
SAVE_PLOTS = True
SAVE_RESULTS = False
SAVE_CSVS = False
```

Then run the relevant script from `python/` (both shown here):

```text
cd python
python ensemble_multistep_medium_md.py
python ensemble_multistep_small_md.py
cd ..
```

Medium-MD reads `results_rev1/` and writes `figures_multistep_rev1/`;
Small-MD reads `results_small_rev1/` and writes `figures_multistep_small_rev1/`.
The scripts leave plotting off by default; the settings above enable it without
refitting or replacing saved forecasts.

The supplied manuscript panels are in [figures_rev1/](figures_rev1/).
Use the [GDP panel map](figures_rev1/README.txt) to select the required panels
and apply the five documented filename aliases.

### AR(1) figures

```text
python python/simulation_ar1_losses.py
```

This reads `results_ar1_rev1/`, writes the four panels to `figures_rev1/ar1/`
and exports `tables_rev1/AR1_loss_regret_summary.csv`, replacing existing exports.
It does not run new simulations.

## 3. Data access and optional regeneration

**Original GDP predictor data are not distributed because of licensing
restrictions.** Obtain them separately for full estimation. See
[data/README.md](data/README.md) for sources, preprocessing requirements and the
folder structure. The prepared CSVs below belong in `data/preprocessed/`;
the DFM benchmark predictions are already supplied.

| Full-estimation exercise | Required inputs |
| --- | --- |
| B=1 Medium-MD or Small-MD | `fullsample_output_data.csv`, `fullsample_output_filled_data.csv`, and the corresponding `data/dfm_medium/` or `data/dfm_small/` predictions |
| B=100 Medium-MD | `fullsample_output_data.csv`, including the extended pre-evaluation sample |
| Equal-budget comparison | The same prepared CSV as B=100, plus `results_b100_rev1/` |

### GDP estimation

For B=1, use the corresponding script from `python/` with
`DO_FORECASTS=True` and `SAVE_RESULTS=True`, retaining the h=1 and K=1000
settings above. This replaces the supplied result files; use a separate working
copy. Details are in the B=1 notes below.

For B=100 and the equal-budget comparison:

```text
python python/ensemble_multistep_medium_md_b100.py --forecast --data data/preprocessed
python python/ensemble_multistep_medium_md_same_budget.py --forecast --data data/preprocessed
```

Run these separately. Outputs go to `results_b100_regenerated/` and
`results_same_budget_regenerated/`, respectively. See `--help` for model
filters and alternative output destinations.

B=100 full estimation can stop at known CV optimizer discrepancies; the separate
recovery tools are not included. This does not affect saved-table reproduction.
See the [B=100 notes](results_b100_rev1/README.txt) for this limitation.

The equal-budget command uses the published frozen 41 × 41 grid; it does not
repeat the historical grid-calibration/production-timing workflow. See the
[same-budget notes](results_same_budget_rev1/README.txt).

### Simulations

No external data are needed. To generate new simulations:

```text
python python/simulation_random_subset.py --simulate
python python/simulation_ar1_losses.py --simulate
```

Outputs go to `results_random_subset_regenerated/` and
`results_ar1_regenerated/`. Full runs also export their tables or figures there;
see `--help` for partial-run options.

## 4. Exercise notes

- B=1: [Medium-MD](results_rev1/README.txt), [Small-MD](results_small_rev1/README.txt), [table definitions](tables_rev1/README.txt)
- [Clipped losses: I.1–I.2](results_clipped_rev1/README.txt)
- [B=100: I.5–I.6](results_b100_rev1/README.txt)
- [Equal-budget comparison: I.7](results_same_budget_rev1/README.txt)
- [Random-subset simulations: J.2–J.3](results_random_subset_rev1/README.txt)
- [AR(1) loss simulations: Figures J.1–J.2](results_ar1_rev1/README.txt)
