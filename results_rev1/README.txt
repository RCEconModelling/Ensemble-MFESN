Medium-MD: revised one-step-ahead results

B=1, K=1000 experts per pool, all 48 quarters from 2008Q1 to 2019Q4.
Final estimation sample: 1990Q1--2007Q4.

The pickle filenames and tuple structure follow the original results/ folder:
  seed: RP; seed_leak: alphaRP;
  seed_lambda: lambdaRP; seed_leak_lambda: alpha-lambdaRP.
Each ensemble tuple contains forecasts, squared forecast errors and expert
MSFEs, with one list entry for h=1. The varying-leak-rate tuples also contain
the leak rate associated with each expert, as in the original alphaRP files.
Expert MSFEs are calculated over all 48 quarters, not the old 41-quarter window.
All saved forecasts and losses use the GDP target's calendar-quarter-end
labels. Original business-quarter-end labels are aligned by quarter only;
forecast and loss values are unchanged.

lambdaRP: 200 draws x 5 ridge penalties.
alpha-lambdaRP: 40 draws x 5 leak rates x 5 ridge penalties.
The grids are alpha={0.1,0.3,0.5,0.7,0.9} and lambda={0.01,0.1,1,10,100}.
The same reservoir/input matrices are used within each draw across the grid.
M-MFESN A RP and alphaRP use the corrected daily-reservoir resampling seed.
The original single-draw benchmarks are unchanged. Original results/ files
are retained and are not overwritten by this package.

From the python/ directory, run ensemble_multistep_medium_md.py with
DO_FORECASTS=False to calculate all four constructions using saved forecasts.
This mode does not require the licensed predictor files. Set DO_PLOTS=True
and SAVE_PLOTS=True to save figures to figures_multistep_rev1/.
The corrected DecHedge/AdaHedge functions are imported from comb_multistep_funcs.py.
Raw squared losses, eta=c0=10000 and RollMSE window r=4 are unchanged.

The prepared forecasts were checked against the revised Table 2 archives.
All eight fixed-grid helper functions additionally passed full regeneration
checks on 2026-10-28: 8,000 experts, 384,000 forecast values, bitwise identical.
manifest.json records the result-file hashes and pool provenance.

To generate forecasts anew, set DO_FORECASTS=True and provide the original
preprocessed predictor data under data/preprocessed/, as in the original code.
SAVE_RESULTS=True writes to results_rev1/; keep a backup if regenerating files.
