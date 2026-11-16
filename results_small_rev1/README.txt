Small-MD: revised one-step-ahead results

B=1, K=1000 experts per pool, all 48 quarters from 2008Q1 to 2019Q4.
Final estimation sample: 1990Q1--2007Q4.
Extended cross-validation sample: 1975Q1--2007Q4.
Small-MD inputs: M1, M4, M5, M7, M11, M12, M14, M15 and D3.

The pickle filenames and tuple structure follow the original results/ folder
and the parallel Medium-MD results_rev1/ package, using the results_S__ prefix:
  seed: RP; seed_leak: alphaRP;
  seed_lambda: lambdaRP; seed_leak_lambda: alpha-lambdaRP.
Each ensemble tuple contains forecasts, squared forecast errors and expert
MSFEs, with one list entry for h=1. The varying-leak-rate tuples also contain
the leak rate associated with each expert, as in the original alphaRP files.
Expert MSFEs are calculated over all 48 quarters, not the old 41-quarter window.
Saved forecasts and losses use the GDP target's calendar-quarter-end labels.
Original business-quarter labels are aligned by quarter only; forecast and
loss values are unchanged. For fresh estimation, the original business-quarter
target dates are retained; the filled S-MFESN target uses calendar-quarter ends.

RP: 1000 draws; alphaRP: 200 draws x 5 leak rates.
lambdaRP: 200 draws x 5 ridge penalties.
alpha-lambdaRP: 40 draws x 5 leak rates x 5 ridge penalties.
The grids are alpha={0.1,0.3,0.5,0.7,0.9} and lambda={0.01,0.1,1,10,100}.
The same reservoir/input matrices are used within each draw across the grid.
M-MFESN A RP and alphaRP use the corrected daily-reservoir resampling seed.
The fixed-parameter tuple contains Mean, AR(1), DFM A, DFM B and the four
single-draw MFESN benchmarks, in the same order as the Medium-MD script.
DFM A/B are the original Small-MD stock/almon models with five factors,
evaluated over all 48 quarters. They are not the Medium-MD benchmarks.

From the python/ directory, run ensemble_multistep_small_md.py with
DO_FORECASTS=False to calculate all four constructions using saved forecasts.
This mode does not require the licensed predictor files. Set DO_PLOTS=True
and SAVE_PLOTS=True to save figures to figures_multistep_small_rev1/.
The corrected DecHedge/AdaHedge functions are imported from comb_multistep_funcs.py.
Raw squared losses, eta=c0=10000 and RollMSE window r=4 are unchanged.
Plot legends use SA. Filenames identify SmallMD and the ensemble construction.

The prepared results reproduce the revised Small-MD Table I.3 and the
corresponding B=1 ECDF/weight-plot data. The original helpers are shared with
ensemble_multistep_medium_md.py. All original results and Medium-MD files are
retained. manifest.json records result hashes, grids and pool provenance.

To generate forecasts anew, set DO_FORECASTS=True and provide the original
preprocessed predictor files under data/preprocessed/, as in the Medium-MD
code. Both fullsample_output_data.csv and fullsample_output_filled_data.csv
are required. The original full-panel availability filter is retained before
selecting the Small-MD columns. DFM files are read from data/dfm_small/,
using stock__5__6__[1]_fixed__2007 and almon__5__6__[1]_fixed__2007.
SAVE_RESULTS=True writes to results_small_rev1/; back up this package first
if regenerating files. SAVE_CSVS=True writes fresh forecasts to exports_small_rev1/.
The existing research environment is required (verified with NumPy 1.26.4
and pandas 2.2.2); no numerical helpers were changed for newer dependencies.
