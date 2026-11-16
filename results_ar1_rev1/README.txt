AR(1) expert losses: Figures J.1 and J.2
=====================================

From the repository root, reproduce all four figure panels and their data:

    python python/simulation_ar1_losses.py

Outputs:
    figures_rev1/ar1/                       Four manuscript figure PDFs
    tables_rev1/AR1_loss_regret_summary.csv  Means, Monte Carlo errors, CIs and bounds

The PDF names match the manuscript, including "exercis" in the name of
the FTL persistence plot. Existing GDP figures are not changed.

Scripts
-------
simulation_ar1_losses.py: main script and saved-results export.
simulation_ar1_funcs.py: simulated losses, weight updates, regret and bounds.
simulation_ar1_plots.py: manuscript-style plots.

These scripts require NumPy, pandas and Matplotlib. They do not
require GDP data, forecast archives, or files outside this repository.

Saved results
-------------
Each q_*.npz contains terminal cumulative regret with axes
(replication, population gap, method), shape (500, 6, 2), plus the q value,
gap grid and method names. All individual replication values are retained;
the scripts calculate the plotted means and confidence intervals from them.
The files do not contain the full loss or weight trajectories.
manifest.json records their SHA256 hashes and the simulation settings.
The original full-trajectory archive hashes identify their provenance.

Design
------
B=500, K=100 and T=1000. Persistence q is 0, .1, .2, .3, .4, .45;
the minimum population gap is .01, .025, .05, .1, .2, .3.
Mean losses are .3 for the best expert, .3+gap for the runner-up,
and .3+gap+.1 for the remaining 98 experts. The common scale is
a=.3*sqrt(.55/(3*1.45)). Losses lie in [0,1] without clipping.

The stationary initial state uses 65 innovations. Replication b uses
PCG64(SeedSequence([2026091701, b, 0])), with b starting at zero.
Innovations are common across design tuples and combination methods,
and independent across experts, time and replications.

FTL assigns equal weights to exact cumulative-loss minimizers.
Decreasing Hedge uses eta_t=2*sqrt(log(K)/t) on past cumulative losses.
Both start with equal weights. There is no AdaHedge in this exercise.
Unlike the GDP experiments, this exercise combines expert losses directly;
it does not compute squared errors of a combined forecast.
Regret subtracts the best expert's cumulative loss in the same replication,
not the loss of the population-best expert.

The pointwise 95% intervals are mean +/- 1.9647389829672648*MCSE, where
MCSE uses the sample standard deviation across the 500 replications.
This retains the original fixed multiplier exactly. It equals the t_498
quantile; the conventional t_499 value is 1.9647293909876649. The difference
in interval half-width is about 0.00049%, with no change to the mean regrets.
The bounds use both factor-of-two mixing envelopes in Appendix J.2,
with individual-loss variance a^2 and excess-loss variance 2*a^2.
Solid lower-panel curves are Hoeffding bounds, dashed curves Bernstein.
Mixing bounds are plotted even at q=0. Figure J.2 also shows the i.i.d.
bounds with hollow circles and diamonds, as specified in its caption.

Recompute from innovations
-------------------------
To regenerate all 36 designs sequentially, with one BLAS thread:

    python python/simulation_ar1_losses.py --simulate

This writes a new results_ar1_regenerated folder, including its own figures
and summary. An existing output folder is not overwritten. For another run,
choose a new folder with --output. Every regenerated terminal regret is
checked against the retained manuscript result (rtol=1e-11, atol=1e-10,
allowing for floating-point library differences). A mismatch stops the run.

For a short check using only the first two replications at q=.45:

    python python/simulation_ar1_losses.py --simulate --q .45 --replications 2 --output results_ar1_check

This partial run saves numerical results only; it is not a manuscript figure.
The full run and saved export were verified with NumPy 1.26.4,
pandas 2.2.2 and Matplotlib 3.9.2.
