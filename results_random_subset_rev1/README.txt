Random-subset experts: Tables J.2 and J.3
======================================

Reproduce the tables (from the repository root)
----------------------------------------------
    python python/simulation_random_subset.py

Equivalently:
    python python/simulation_tables_random_subset.py

These commands use only the compact saved losses in this folder. They do
not simulate data or fit experts. They write three CSVs to tables_rev1/:
    TableJ2_random_subset_oracle.csv
    TableJ3_random_subset_no_oracle.csv
    random_subset_mean_losses.csv

The first two contain all manuscript entries at full precision, with one
row per panel, DGP and training length, seven combination columns, and
best/second-best labels. Display the numbers to three decimal places to
match the manuscript. Rankings use unrounded values, so equally displayed
numbers can have different rankings. The third CSV reports the numerator,
denominator and number of pooled replications for every table entry.

Design and reporting
--------------------
There are 30 predictors and two Gaussian AR(1) predictor specifications:
baseline rho = {0.1,0.5,0.9} and robustness rho = {0.05,0.25,0.45}, with
10 predictors at each value. Predictor innovation variances are
v_X*(1-rho^2), where v_X=4/(1-0.9^2). Predictors start at zero, and 200
pre-sample observations are discarded. Y_t is the mean of the 30 predictors
plus independent N(0,1) noise. Forecasts use the contemporaneously available
X_t. No empirical GDP dataset is required.

Each pool contains p random OLS experts, p in {10,100,200}. Training lengths
are 100,300,500, followed by 1,000 evaluation observations. Each OLS expert
has its own estimated intercept; coefficients stay fixed over evaluation.
Small subsets contain 5 predictors; large subsets contain 29. Mixed pools
contain p/5 experts at each size 5,10,15,20,25, in interleaved size order.
Predictors are sampled without replacement within an expert. The original
generator also rejects duplicate subsets across experts, except in large
pools with p=100 or 200, where only 30 distinct size-29 subsets exist.

The oracle uses all 30 predictors and their population coefficients 1/30,
with zero intercept; it is not an estimated full OLS regression. Table J.2
includes it (K=p+1). Table J.3 removes it from the same pool (K=p), keeping
the random experts unchanged.

There are B=500 replications per design: 27 regime/p/training settings x
2 DGPs x 2 oracle scenarios = 108 designs. The two oracle scenarios share
the 27,000 simulated pools, giving 54,000 scenario replications.
For zero-based replication b, data seed=12345+b and expert seed=1012345+b.
The same seed schedule is reused across design cells, as in the original
exercise. Do not replace it with independent new seeds if reproducing the
reported numbers. All predictor innovations are drawn before response noise.

Squared errors are used for both weight updates and evaluation; there is
no clipping or learning-rate scaling in these two tables. The corrected
combination routines are imported from comb_multistep_funcs.py unchanged:
Hedge eta=0.5, DecHedge c0=2, and RollMSE r=4 with epsilon=1e-9 and equal
weights until four losses are observed. Online weights use only past losses.
FTL retains the helper's numerical tie convention; AdaHedge uses its
corrected mixability-gap update. SA and Median are forecast combinations.

Each table entry is mean(method MSFE)/mean(realized-best-expert MSFE), NOT
the mean of replication-specific ratios. The best expert is determined
ex post within each replication and oracle scenario. Panel A pools all
three subset regimes and all three p values (4,500 replications per column);
Panels B--D pool the three p values within a regime (1,500 per column).
DGP and training length stay fixed within each column. A ratio below one
means the combination has lower mean MSFE than this realized-best benchmark.

Compact files
-------------
g01.npz through g27.npz follow this loop order:
    regime: small_only, mixed, large_only
    p: 10,100,200
    T_train: 100,300,500
The manifest also maps every filename to its design.

Each file contains method_msfe[2,2,500,7], best_msfe[2,2,500], design and
scenario labels, method names, training/test sizes, p, regime and both seed
sequences. The axes are DGP, oracle scenario, replication and method.
No full forecast matrices, weights or simulated datasets are included.
manifest.json records file hashes, settings and original source hashes.
The archives retain the verified raw losses, with the revised r=4 RollMSE
and corrected DecHedge/AdaHedge results. The other methods are unchanged.

Run the simulation again (optional)
----------------------------------
    python python/simulation_random_subset.py --simulate

The main script generates data and subsets, fits each expert, and computes
all seven combinations. New outputs go to results_random_subset_regenerated/;
the supplied results are never overwritten. The run checks each pool's
MSFEs against the supplied reference and stops on numerical disagreement.
It uses one worker and one BLAS thread. Completed grid files can be reused
after checking the code/configuration and numerical results. An incomplete
grid must be rerun. Do not remove run.lock while its process is active.

A small check, without launching the full B=500 run:
    python python/simulation_random_subset.py --simulate --grid 1 --replications 2 --output results_random_subset_check

--grid selects one of the 27 settings; omitting it runs all settings.
--replications uses the first specified replications of the original seed
schedule. Partial checks do not export manuscript tables. Use a new output
folder when changing the replication count or code. After all 27 grids
finish with B=500, the driver exports tables to its output's tables/ folder.

Only NumPy and pandas are needed. The reconstruction checks use NumPy 2.3.5
and pandas 3.0.1, matching the retained data/OLS reconstruction environment.
Different BLAS or library versions can introduce numerical differences;
the supplied loss checks must pass rather than assuming exact reproduction.
The corrected combination results were independently checked in the earlier
study. The repository's bounded tests regenerate the first and last
replication in every grid and both DGPs, not all 27,000 pools afresh.

Implementation
--------------
simulation_random_subset.py: sequential driver, checkpoints and saved-table mode.
simulation_random_subset_funcs.py: data, subsets, OLS, combination calls and losses.
simulation_tables_random_subset.py: compact-result checks and table calculations.

Existing repository helpers, earlier results and manuscript files are unchanged.
