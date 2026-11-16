"""
Random-Subset Simulation Functions

Linear Gaussian DGP, random-subset OLS experts and forecast combinations.
Retains the original simulation draws and uses the corrected combination helpers.
"""

import numpy as np
import pandas as pd
from comb_multistep_funcs import (
    median_forecast, simple_average_forecast, rollMSE_weights_forecast,
    ftl_forecast, const_hedge_forecast, dec_hedge_forecast, adahedge_forecast,
)

#region // GLOBAL SETTINGS ------------------------------------

METHODS = ['Median', 'SA', 'RollMSE', 'FTL', 'Hedge', 'DecHedge', 'AdaHedge']
DESIGNS = ['baseline', 'robustness']
SCENARIOS = ['oracle', 'no_oracle']
REGIMES = ['small_only', 'mixed', 'large_only']
GRID = [(regime, p, n) for regime in REGIMES
        for p in [10, 100, 200] for n in [100, 300, 500]]
B = 500
J = 30
T_TEST = 1000
BURNIN = 200
BASE_SEED = 12345
ETA = .5
C0 = 2.

#endregion

#region --- Generate Data and Experts


def generate_data(design, n, b):
    # Draw all predictor innovations before response innovations, as originally.
    if design not in DESIGNS:
        raise ValueError('Unknown DGP specification.')
    rho = np.array([.1, .5, .9])
    variance = 4. / (1. - rho[2]**2)
    if design == 'robustness':
        rho = rho / 2.
    scales = np.repeat(np.sqrt(variance * (1. - rho**2)), 10)
    rho = np.repeat(rho, 10)
    rng = np.random.default_rng(BASE_SEED + b)
    length = n + T_TEST + BURNIN
    innovations = rng.normal(0., scales, size=(length, J))
    noise = rng.normal(0., 1., size=length)
    X = np.zeros((length, J))
    for t in range(1, length):
        X[t] = rho * X[t-1] + innovations[t]
    y = X.mean(axis=1) + noise
    return X[BURNIN:], y[BURNIN:]


def draw_subsets(regime, p, b):
    # Sample without replacement within each expert. Mixed sizes are interleaved.
    if regime == 'small_only':
        sizes = np.full(p, 5)
    elif regime == 'mixed':
        sizes = np.tile([5, 10, 15, 20, 25], p // 5)
    elif regime == 'large_only':
        sizes = np.full(p, 29)
    else:
        raise ValueError('Unknown subset regime.')
    rng = np.random.default_rng(BASE_SEED + 1000000 + b)
    # Only 30 distinct subsets of size 29 exist; retain the original convention.
    unique = not (regime == 'large_only' and p > J)
    used = set()
    subsets = [np.arange(J)]
    for size in sizes:
        while True:
            subset = np.sort(rng.choice(J, size=int(size), replace=False))
            key = tuple(subset)
            if not unique or key not in used:
                break
        used.add(key)
        subsets.append(subset)
    return subsets


def fit_experts(X, y, subsets, n):
    # Column zero is the population oracle, not an estimated full regression.
    forecasts = np.zeros((T_TEST, len(subsets)))
    forecasts[:, 0] = X[n:] @ np.full(J, 1. / J)
    for k, subset in enumerate(subsets[1:], 1):
        train = np.column_stack([np.ones(n), X[:n, subset]])
        test = np.column_stack([np.ones(T_TEST), X[n:, subset]])
        beta, _, rank, _ = np.linalg.lstsq(train, y[:n], rcond=None)
        if rank < train.shape[1]:
            raise np.linalg.LinAlgError(f'Expert {k} is rank deficient.')
        forecasts[:, k] = test @ beta
    if not np.isfinite(forecasts).all():
        raise ValueError('Non-finite expert forecast.')
    return forecasts

#endregion

#region --- Forecast Combination and Evaluation


def combine_forecasts(forecasts, y):
    losses = (forecasts - y[:, None])**2
    median, _ = median_forecast(forecasts)
    average, _ = simple_average_forecast(forecasts)
    rolling, _ = rollMSE_weights_forecast(forecasts, y[:, None], h=1, window_size=4)
    ftl, _ = ftl_forecast(forecasts, losses, h=1)
    hedge, _ = const_hedge_forecast(forecasts, losses, h=1, learning_rate=ETA)
    dec, _ = dec_hedge_forecast(forecasts, losses, h=1, c0=C0)
    ada, _ = adahedge_forecast(forecasts, losses, h=1)
    combined = pd.concat([median, average, rolling, ftl, hedge, dec, ada], axis=1)
    combined.columns = METHODS
    if not np.isfinite(combined.to_numpy()).all():
        raise ValueError('Non-finite combination forecast.')
    return combined.to_numpy()


def evaluate_pool(forecasts, y):
    # Same random experts with and without the population oracle.
    methods, best = [], []
    for start in [0, 1]:
        experts = forecasts[:, start:]
        combined = combine_forecasts(experts, y)
        methods.append(np.mean((combined - y[:, None])**2, axis=0))
        best.append(np.mean((experts - y[:, None])**2, axis=0).min())
    return np.asarray(methods), np.asarray(best)

#endregion
