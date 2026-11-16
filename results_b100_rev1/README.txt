Medium-MD: B=100 expert resampling
================================

Scope
-----
Four MFESN models x four ensemble constructions x 100 independent expert
pools, K=1000, h=1. These are the revised Table I.5 results and the companion
CV OOS rank / win-rate table (Table I.6). This is not the same-budget exercise.
Small-MD was evaluated for B=1 and is not included here.

Reproduce the two tables (from the repository root)
-------------------------------------------------
python python/ensemble_tables_b100.py

Equivalently:
python python/ensemble_multistep_medium_md_b100.py

Neither command fits models. Both read only the compact results in this
folder and export the following files to tables_rev1/:
  TableI5_medium_MD_B100.csv
  TableI6_medium_MD_B100_ranks_wins.csv
  medium_MD_B100_paired_losses.csv

Table I.5: CV is mean CV-selected-expert MSFE divided by the in-sample-mean
benchmark MSFE. All seven combination columns divide the method's mean
MSFE by the CV-selected expert's mean MSFE, across the same 100 pools.
These are ratios of means, not means of pool-specific ratios.

Table I.6: CV OOS rank is 1 plus the number of pool experts with strictly
smaller MSFE; exact ties share a rank. Quartiles use linear interpolation.
The display column rounds ranks to integers (half upwards); separate
numeric columns preserve unrounded quartiles. Win rates count strictly
negative paired MSFE differences, with no tie tolerance; B=100 gives integer
percentages. The paired-loss file gives these differences for every pool.
Expert indices are zero-based; replication labels are one-based.

Saved files
-----------
Each model/construction NPZ contains 100 pools' expert CV scores, assigned
or tuned ridge penalties, expert OOS MSFEs, selected expert indices and
forecasts, seven combination forecasts, observed GDP, and saved pool-level
MSFEs. The exporter also checks losses recomputed from the saved paths.
Floating-point reductions are retained from the verified run.

reservoir_seeds.npz retains the exact seed schedule used in those runs:
100 x draws x frequencies, uint64. S-MFESNs have one seed; M-MFESNs have
monthly/daily seed pairs. Each reservoir's A and C calls use the same seed.
The same matrices are used within a draw across leak rates and penalties.
The schedules, including the original nested expansions, must not be
replaced by a newly chosen master seed if reproducing the reported tables.

manifest.json lists settings and SHA256 hashes of the compact archives.
No raw predictor dataset, full expert forecast matrices or aggregation
weights are supplied here. The compact results reproduce both tables but
cannot recompute combination weights. Refitting requires the data below.

Generate the pools again (optional, expensive)
---------------------------------------------
python python/ensemble_multistep_medium_md_b100.py --forecast

Requires data/preprocessed/fullsample_output_data.csv as used in the main
Medium-MD script, with the extended pre-test observations. A different
location can be passed with --data. Existing data-access restrictions apply.
New results are written to results_b100_regenerated/, never this folder.
Use --output to select another new folder. Individual cases can be run with
--model smfesn_a (or smfesn_b/mmfesn_a/mmfesn_b) and --ensemble RP (or
alphaRP/lambdaRP/alpha-lambdaRP). Run cases sequentially, not concurrently.
Completed pools are reused after checking the configuration and seeds.
Interruptions within a pool require that pool to be fitted again.
After all 16 cases complete, tables are exported to the new output's tables/.

The driver uses one worker and one BLAS thread. The reference runtime used
NumPy 1.26.4, pandas 2.2.2 and SciPy 1.14.0. Different numerical libraries can
change optimizer paths; compare against the supplied results, rather than
assuming bitwise reproduction across environments.

Implementation
--------------
The driver loops over cases and pools. esn_multistep_resampling_funcs.py
prepares the original data split, builds reservoirs with explicit seeds,
and captures the original CV objective. The original ESN, final fitting,
forecasting and corrected combination routines are imported unchanged.
RP/alphaRP tune lambda using the original isotropic CV optimizer. lambdaRP
and alpha-lambdaRP evaluate that same objective at the five fixed penalties
without optimization. CV selects one expert in every construction.
The original M-MFESN CV audit scores the optimizer's returned penalty and
records known abnormal line-search exits without retuning; unexplained
objective discrepancies raise an error. Optimizer status is saved per pool.

Numerical caveat and full-regeneration limitation
-----------------------------------------------
Independent full-design verification of the original exercise found 16 unique
S-MFESN cases in which an unsuccessful optimizer returned a reported CV
objective different from the objective evaluated at its returned penalty.
These differences are NOT roundoff. The original S-MFESN exercise retained
the optimizer-reported scores and returned penalties; those historical values
are retained here, not replaced by returned-point scores. Separately, 11
pre-existing M-MFESN helper reconciliations use the returned-point scoring
convention described above. These are distinct sets of cases.

The full-design verification retained the original scoring conventions and
confirmed the forecast paths, selected experts, ranks and strict wins, with
unchanged printed Tables I.5/I.6. This agreement does not establish optimizer
convergence or that every reported CV score equals the objective at its
returned penalty.

The calculation scripts in this release are unchanged. In particular, the
full-estimation driver still stops on an S-MFESN reported/returned-point
objective discrepancy. The separately used exact-case diagnosis, guarded
recovery and checkpoint-verification tools are not distributed here. The
saved-result commands above reproduce both tables without those tools, but
the optional full-estimation command is not an unattended end-to-end replay
of that separately verified run. Do not loosen its checks, retune failed
cases, or silently substitute scores.

Validation: 1975Q1-2007Q4, ten expanding folds, five validation quarters per
fold. Final estimation: 1990Q1-2007Q4. Evaluation: all 48 quarters,
2008Q1-2019Q4. Selected expert and fitted coefficients stay fixed.
Raw squared losses, eta=c0=10000, RollMSE window=4, corrected DecHedge and
AdaHedge. All data observations remain fixed across the 100 replications;
only the reservoir draws change.
