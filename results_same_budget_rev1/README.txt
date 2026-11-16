Medium-MD same-budget single-model comparison (Table I.7)
======================================================

This folder contains the compact saved results for the approximately
matched computational-budget comparison. It uses the corrected DecHedge
and AdaHedge combinations and all 48 quarters, 2008Q1--2019Q4.
It is a Medium-MD exercise only.

Reproduce the table
-------------------
From the repository root, run:

    python python/ensemble_multistep_medium_md_same_budget.py

This uses only the saved results in this folder and results_b100_rev1;
it does not refit experts, read the raw data or measure new runtimes.
NumPy and pandas are required. It writes:

    tables_rev1/TableI7_medium_MD_same_budget.csv
    tables_rev1/medium_MD_same_budget_paired_losses.csv

The first file reports the same-budget single model (EB) relative to the
in-sample mean, and all seven combinations relative to the same-budget
model. These are ratios of mean MSFEs across the 100 pools, not means of
within-pool ratios. The second file contains the corresponding paired
pool-level MSFEs and differences. Best and second-best combination
labels are supplied in the first file; EB is not included in the ranking.

The compact B=100 package supplies the ensemble results; there is no
second copy here. The same-budget single model is different from the
CV-selected expert drawn from the 1,000-member pool in Tables I.5/I.6.

Design
------
There are 100 pools for each of S-MFESN A, S-MFESN B, M-MFESN A and
M-MFESN B. Each alpha-lambdaRP ensemble crosses 40 reservoir draws with
five leak rates and five ridge penalties, giving 1,000 experts.
The original grids are alpha = {0.1,0.3,0.5,0.7,0.9} and
lambda = {0.01,0.1,1,10,100}.

The same-budget model uses only the first draw (monthly/daily pair for
M-MFESN) of the same pool. It searches a 41 x 41 grid, equally spaced in
alpha on [0.1,0.9] and log10(lambda) on [-2,2], retaining the five original
anchors in each dimension. The same scalar alpha and lambda are assigned
to both M-MFESN reservoirs. Selection minimizes the original ten-fold
pre-evaluation CV objective; exact ties use the first row-major minimum.

CV uses the extended 1975Q1--2007Q4 sample, with five validation quarters
per fold. The original helper sums the ten fold MSFEs; division by ten
does not affect selection. Final estimation uses 1990Q1--2007Q4, and
evaluation uses 48 quarters, 2008Q1--2019Q4. Expert parameters remain fixed
over evaluation. No evaluation losses enter grid choice or selection.
Raw squared-error losses, eta=c0=10000 and all seven original combination
paths are retained (with the corrected DecHedge/AdaHedge routines).

The 41 x 41 grids were frozen after the historical time-only calibration
and stability checks, before production. The scripts here reproduce that
published frozen-grid design; they do not implement a new grid search for
another computer's time budget. Both arms use their own reservoir states.
Exact-input state/alignment results are reused within each draw and alpha
as in the original timing implementation, without changing the original
fitting or CV calculations.

Saved files
-----------
smfesn_a.npz, smfesn_b.npz, mmfesn_a.npz, mmfesn_b.npz:
    selected_forecasts: 100 x 48, one selected forecast per pool;
    cv_objective_sum: 100 x 41 x 41, complete pre-test CV objectives;
    selected_indices: 100 x 2, zero-based alpha/lambda indices;
    first_seeds: 100 x 1 or 100 x 2, original first-draw seeds;
    alpha_grid, log10_lambda_grid, observed, in_sample_mean_forecast,
    pool_ids and quarters.

manifest.json:
    SHA256 checks for the numerical/timing files and the five required
    B=100 dependency files. Numerical and timing files are checked before
    table export or new numerical work.

timing_history.json:
    Individual original and corrected timing measurements, passed gates,
    rejected/repeated attempts, recovery records, original production
    block records, frozen-grid gate and source hashes. Paths in this file
    are logical historical identifiers, not files to load from this repo.
    Full historical expert arrays and controller logs are not included.

timing_summary.json:
    Recalculated paired ratios/spreads, production-block checks and runtime
    totals from the retained records. Recheck without running forecasts:

    python python/ensemble_same_budget_timing.py

Timing scope and limitations
----------------------------
This is representative runtime matching, NOT 400 contemporaneously timed
ensemble/model pairs. Each representative check measured both arms on
original pools 1,2,3 (zero-based), with two repeats, interleaved ensemble /
tuning then tuning / ensemble. Every pool required a tuning/ensemble
repeat-median ratio in [0.9,1.1] and both repeat spreads at most 20%, with
gross wall/CPU and recorded power-transition checks. Grids were not resized
after freezing. Original production used ten-pool block checks against the
accepted runtime epoch, refreshing both representative arms and remeasuring
affected blocks when needed, without changing forecasts or grids.

After correcting the aggregation routines, both arms were measured freshly
on the three representative pools for each model. The original 400 selected
forecasts and production timings were retained; all 400 production timings
were NOT remeasured after the correction.

The history retains 527 original and 108 corrected completed timing units.
Original recorded numerical wall time totals about 13.0284 hours, including
9.5326 hours for the accepted production units and 3.4958 hours for other
retained calibration/reference/repeated units. The original logged gross
span is at least 14.7978 hours; it is a lower bound, not an exact end-to-end
project runtime. The corrected stage contains about 1.6710 numerical wall
hours and took about 2.6396 hours between the controller's stage markers,
including waiting and other overhead. The records disclose timing-only
recovery/cooldowns and the original path-length repair. A partial unit with
saved arrays but no completed timing cannot contribute a measured duration.
No unobserved cause of runtime or power changes is inferred. These costs
are not the accepted numerical budget alone.

Optional numerical reproduction
-------------------------------
The raw input data and the original fitting dependencies are needed only
for refitting. As for the B=100 script, supply the directory containing
fullsample_output_data.csv. The validated legacy environment uses NumPy
1.26.4, pandas 2.2.2 and SciPy 1.14.0; see the original repository
requirements for the other fitting/plotting dependencies. Numerical
identity across different library/BLAS versions is not guaranteed.

    python python/ensemble_multistep_medium_md_same_budget.py --forecast --data PATH_TO_DATA

This refits the 400 selected models at the frozen grids. It checks the full
CV matrices, selected indices, grids, seeds and forecasts against the saved
results. It does NOT certify a new matched production budget. Outputs go
to results_same_budget_regenerated, never the supplied result folders.
Completed pool files can be reused only after exact numerical checks and
matching data/code configuration. A failed numerical check stops the run.

Use --model smfesn_a (or smfesn_b, mmfesn_a, mmfesn_b) for one specification;
--output PATH selects a separate destination. Both numerical modes use one
worker and set OMP/MKL/OpenBLAS thread counts to one before NumPy imports.
Do not run them concurrently with another numerical job. A run.lock guards
the chosen destination; never remove it while its process is active.

To measure fresh representative pairs at the SAME frozen grids:

    python python/ensemble_multistep_medium_md_same_budget.py --check-timing --data PATH_TO_DATA --output results_same_budget_timing_new

This includes expert generation/fitting/forecasting and all seven
aggregators in the ensemble arm, excluding CV used only for the selected-
expert benchmark. The tuning arm includes full CV, deterministic selection,
final fitting and forecasting. Shared preprocessing, warm-up, checkpoints,
validation and output are outside numerical timing. These refactored
wrappers can have different overhead from the historical controller;
fresh measured checks, not the historical target, determine their outcome.

Fresh checks retain all completed timings, numerical diagnostics and
separate run directories. The compact archive permits exact selected-model
identity checks and ensemble MSFE/combined-path checks, but not comparison
of every individual expert forecast to a stored full historical matrix.
Power state is observed on Windows without changing settings; unavailable
telemetry does not yield a passed gate. A failed gate is retained and stops
the command; this concise reproduction script does not run the historical
automatic cooldown/recovery controller. Do not resize the grids or loosen
tolerances to obtain a pass. Neither this command nor --forecast replaces
the historical ten-pool production timing protocol.

Original results, helper functions and manuscript files are not modified.
