Medium-MD: B=1 quantile-clipped combination losses
================================================

Tables I.1 and I.2 use the same 16 pools as Table 2: four MFESN models
crossed with RP, alphaRP, lambdaRP and alpha-lambdaRP. Each pool has K=1000
experts. Forecasting is one-step-ahead over 48 quarters, 2008Q1-2019Q4.

Run from the repository root:

    python python/ensemble_multistep_medium_md_clipped.py

Or, from python/:

    python ensemble_multistep_medium_md_clipped.py

This script reads only results_rev1/ and calls the existing corrected
functions in comb_multistep_funcs.py. No expert is refitted. The original
predictor files are not needed. It uses NumPy, pandas and rich from the
existing environment (verified with NumPy 1.26.4 and pandas 2.2.2).
SAVE_RESULTS and SAVE_CSVS control the new result and table exports.
Rerunning replaces only this exercise's generated files; retain a copy
first if changing its settings. No existing main script, combination
helper, forecast archive or Table 2/I.3 CSV is changed.

Loss construction and learning rates
-----------------------------------
The script contains the complete threshold calibration, loss construction,
combination and table-export steps. Thresholds are the linearly interpolated
95th and 99th percentiles of SQUARED differences between 20,000 independently
resampled pairs of training GDP observations (1990Q1-2007Q4, n=72).
NumPy default_rng uses seed 20261001, drawing all left indices first and
then all right indices. One set of pairs is shared by every specification.
No evaluation observations enter calibration.

For each expert, raw_loss = (forecast - observed GDP)^2.
For combination, clipped_loss = minimum(raw_loss / q_p, 1), where q_p is
the calibrated squared threshold. Evaluation always uses unclipped squared
errors of the combined forecasts. FTL, Hedge, DecHedge and AdaHedge use the
same repository helpers as Table 2; only their loss inputs change, together
with the following scaling for Hedge and DecHedge:

    eta_p = 10000 * q_p
    c0_p  = 10000 * q_p

The raw comparison uses eta=c0=10000. Earlier unscaled-learning-rate
diagnostics are not part of this package. Median, SA and RollMSE are not
reported in Tables I.1/I.2 and are not recomputed here.

Saved data
----------
calibration.json records the periods, seed, quantile method, thresholds,
learning-rate parameters and in-sample-mean MSFE denominator.

results_M__clipped_losses.pkl is a dictionary containing:
  calibration:       the same calibration settings and values;
  calibration_draws: training GDP, zero-based left/right sampled indices,
                     and the 20,000 squared pair differences;
  observed_gdp:      the 48 evaluation observations;
  pools:             nested dictionaries by ensemble and model name.

For example, pools['alphaRP']['M-MFESN A'] contains:
  source_file:          the existing expert-forecast archive;
  combination_losses:  Raw/Raw, 95/Raw, 99/Raw expert-loss DataFrames,
                       each 48 quarters x 1000 experts;
  combined_forecasts:   the same three treatments, each 48 quarters x
                       four methods (FTL, Hedge, DecHedge, AdaHedge);
  evaluation_losses:   the corresponding UNCLIPPED squared-error DataFrames.

Expert column order is retained from the source archive. Combined forecasts
and losses are stored, but the large weight matrices are not duplicated.
Only load trusted pickle files.

Tables
------
The same script prints both tables and writes:
  tables_rev1/TableI1_medium_MD_clipped95.csv
  tables_rev1/TableI2_medium_MD_clipped99.csv

Each table has 64 rows: four constructions x four models x four methods.
MSFE is mean evaluation loss across the 48 quarters. Relative_MSFE divides
by the in-sample-mean forecast's MSFE. Raw_MSFE and Raw_Relative_MSFE are the
matching no-clipping results. Change_vs_raw_pct = 100*(MSFE/Raw_MSFE - 1).
Thus -6.15 means -6.15%, not -615%. Negative values indicate lower MSFE
than the same method without clipping. This is NOT the percentage change
relative to the in-sample mean reported in the Table 2 CSV.

CSV values retain full numerical precision. The console/manuscript display
uses three decimal places for relative MSFE and two for percentage changes.

Verification (2026-10-29)
-------------------------
All 128 clipped-method results and percentage changes reproduce the
verified clipping archive, and all 256 printed numerical entries match
Tables I.1/I.2 in the manuscript. The 64 raw-method results match Table 2.
The sampled pairs, all expert-loss matrices, combined forecasts and
recomputed combination weights match the retained archive exactly.
A second run produced byte-for-byte identical JSON, pickle and CSV files.
