B=1 GDP results tables
=====================

Table2_medium_MD.csv: Table 2, Medium-MD.
TableI3_small_MD.csv: Table I.3, Small-MD.

Both files use B=1, K=1000 experts per pool and h=1, evaluated over all
48 quarters, 2008Q1-2019Q4. They include RP, alphaRP, lambdaRP and
alpha-lambdaRP, with the corrected DecHedge and AdaHedge routines.
The loss is unclipped squared error. eta=c0=10000 and RollMSE r=4.
These are the main B=1 tables, not the clipped-loss or B=100 exercises.

Contents
--------
Each CSV contains 132 records: four common benchmarks, followed by
four ensemble constructions x four MFESN specifications x eight methods
(the single-draw Baseline plus the seven forecast combinations).
Rows follow the manuscript's order. Baseline values repeat across the four
constructions, as in the manuscript; these are not additional replications.

Ensemble: Benchmark, RP, alphaRP, lambdaRP or alpha-lambdaRP.
Model:    Benchmark name or MFESN specification.
Method:   Baseline, Median, SA, RollMSE, FTL, Hedge, DecHedge or AdaHedge.
MSFE:     Mean of the 48 unclipped squared forecast errors.
Relative_MSFE:
          MSFE divided by the in-sample-mean forecast's MSFE over the same
          48 quarters. The denominator is 4.701649578042202e-05.
Change_vs_mean_pct:
          100 * (Relative_MSFE - 1). Negative values mean lower MSFE than
          the in-sample-mean forecast. Values are in percent units: -6.15
          means -6.15%, not -615%.
Combination_rank:
          Rank among the seven combinations for the same construction and
          model, using unrounded MSFE. 1 is best and 2 is second-best.
          Blank for Baseline rows, which are excluded from this ranking.

Precision and manuscript comparison
-----------------------------------
CSV values retain full numerical precision. For manuscript presentation,
round Relative_MSFE to three decimal places and Change_vs_mean_pct to two.
Do not calculate percentage changes from the already-rounded relative MSFE.
CSV has no bold or underline formatting; Combination_rank identifies the
corresponding manuscript emphasis. The four common benchmarks also have
percentage changes in the CSV, although the manuscript shows only their
relative MSFEs.

Checked on 2026-10-29 against the current manuscript: all 264 printed
relative MSFEs, all 256 printed percentage changes and all 32 best/second
combination pairs agree at the manuscript's reporting precision.

Sources and reproduction
------------------------
Medium-MD: python/ensemble_multistep_medium_md.py, results_rev1/.
Small-MD:  python/ensemble_multistep_small_md.py, results_small_rev1/.

To regenerate both CSV files, run from the repository root:

    python python/ensemble_tables_b1.py

Or, from the python/ directory:

    python ensemble_tables_b1.py

The exporter calls the same functions in comb_multistep_funcs.py, with
the same settings as the two main scripts, and recomputes MSFEs from the
saved forecasts. It needs Python, NumPy and pandas (verified with NumPy
1.26.4 and pandas 2.2.2). The original predictor data are not needed.
Only the two table CSVs are replaced; models are not refitted, no figures
are generated and saved forecasts are not changed. No main-script flags
need to be edited. To retain the existing CSVs when checking a new export:

    python python/ensemble_tables_b1.py --output-dir tables_check

This last command is run from the repository root. The alternate output
directory is relative to the directory from which the command is run.

Important: the existing console's percentage rows compare each combination
with its corresponding single-draw MFESN baseline. The manuscript instead
compares with the in-sample mean. These CSVs use the manuscript definition
above; the main scripts' console output has not been changed. SAVE_CSVS
exports forecast paths in the original workflow, not these consolidated
tables. The main scripts, combination helpers, saved results and manuscript
are unchanged by this exporter.
