"""
Forecast Combination with Simulated AR(1) Losses

Data generation, FTL and decreasing Hedge, and regret bounds.
"""

import hashlib
import json
from math import log, sqrt

import numpy as np
import pandas as pd

#region // GLOBAL SETTINGS ------------------------------------

METHODS = ['ftl', 'decreasing_hedge']
Q_GRID = [0., .1, .2, .3, .4, .45]
DELTA_GRID = np.array([.01, .025, .05, .1, .2, .3])
B, K, T = 500, 100, 1000
MU_STAR, SEPARATION = .3, .1
A = .3 * (.55 / (3 * 1.45))**.5
C0 = 2.
MASTER_SEED = 2026091701
STATIONARY_TERMS = 65
# Retain the fixed 95% confidence-band multiplier used in the manuscript run.
CI_CRITICAL = 1.9647389829672648

#endregion

#region --- Simulated Losses and Combinations


def simulate_losses(q, replications=B):
    """Return terminal regret, using the same innovations for every design."""
    means = np.empty((len(DELTA_GRID), K))
    means[:, 0] = MU_STAR
    means[:, 1] = MU_STAR + DELTA_GRID
    means[:, 2:] = MU_STAR + (DELTA_GRID + SEPARATION)[:, None]
    regret = np.empty((replications, len(DELTA_GRID), len(METHODS)))

    # Batches keep memory use small; replications remain independent.
    for first in range(0, replications, 50):
        last = min(first + 50, replications)
        innovations = np.empty((last-first, STATIONARY_TERMS+T, K))
        for i, b in enumerate(range(first, last)):
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([MASTER_SEED, b, 0])))
            innovations[i] = rng.uniform(-sqrt(3), sqrt(3), size=(STATIONARY_TERMS+T, K))
        scale = np.sqrt(1-q*q)
        z = scale * np.einsum('j,bjk->bk', q**np.arange(STATIONARY_TERMS),
                             innovations[:, :STATIONARY_TERMS])
        cumulative = np.zeros((last-first, len(DELTA_GRID), K))
        combined = np.zeros((last-first, len(DELTA_GRID), len(METHODS)))

        for t in range(1, T+1):
            z = q*z + scale*innovations[:, STATIONARY_TERMS+t-1]
            losses = means[None, :, :] + A*z[:, None, :]
            if losses.min() < 0 or losses.max() > 1:
                raise ValueError('Generated losses are outside [0, 1].')

            # Weights for round t depend only on losses through t-1.
            leaders = cumulative == cumulative.min(axis=-1, keepdims=True)
            ftl = leaders / leaders.sum(axis=-1, keepdims=True)
            eta = C0 * np.sqrt(np.log(K)/t)
            hedge = np.exp(-eta*(cumulative-cumulative.min(axis=-1, keepdims=True)))
            hedge /= hedge.sum(axis=-1, keepdims=True)
            combined[:, :, 0] += (ftl*losses).sum(axis=-1)
            combined[:, :, 1] += (hedge*losses).sum(axis=-1)
            cumulative += losses

        # Best expert in hindsight, separately for each replication.
        regret[first:last] = combined - cumulative.min(axis=-1)[:, :, None]
        print(f'q={q:.2f}: {last}/{replications}', flush=True)
    return dict(regret=regret, q=np.array(q), delta_grid=DELTA_GRID,
                methods=np.array(METHODS))

#endregion

#region --- Expected Regret Bounds


def regret_bounds(q, delta):
    """Mixing and i.i.d. bounds plotted in Figures J.1 and J.2."""
    l = log(K)
    variance = A*A
    excess_variance = 2*variance
    epsilon = (1-2*q)/(1-q)
    r = q/(1-q)
    r_bar = 1-epsilon*epsilon

    # Both dependence envelopes include the factor 2.
    tilde_phi = 1+8*r/(1-r)
    phi = 1+sqrt(2)*sqrt(r)/(1-sqrt(r))
    tilde_phi_bar = 1+8*r_bar/(1-r_bar)
    phi_bar = 1+sqrt(2)*sqrt(r_bar)/(1-sqrt(r_bar))

    ftl = dict(
        bound_H=3+8*tilde_phi*(l+4)/delta**2,
        bound_B=2+4*phi**2*(32*variance+delta)*(log(2*K)+2)/delta**2,
        bound_iid_H=2+(2*l+4)/delta**2,
        bound_iid_B=2+(8*variance+4*delta/3)*(log(2*K)+2)/delta**2,
    )
    hedge = dict(
        bound_H=2+sqrt(l)+(4*sqrt(2*tilde_phi_bar)*l
                          +4*tilde_phi_bar*(log(K-1)+1.5))/delta
                          +16*(4*tilde_phi_bar+1)/delta**2,
        bound_B=1+sqrt(l)+(2*sqrt(33)*phi_bar*l
                          +4*tilde_phi_bar*(log(K-1)+1.5))/delta
                          +(4*phi_bar**2*(32*excess_variance+delta)+16)/delta**2,
        bound_iid_H=(4*delta*l+25)/delta**2,
        bound_iid_B=1+sqrt(l)+((4*sqrt(2/3)+1)*delta*l
                              +8*(excess_variance+delta/3)+16)/delta**2,
    )
    return dict(ftl=ftl, decreasing_hedge=hedge)

#endregion

#region --- Saved Results


def verify_results(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Saved result changed: {name}')


def summarize_results(folder):
    """Recompute means, Monte Carlo errors, confidence intervals and bounds."""
    rows = []
    for q in Q_GRID:
        with np.load(folder / f'q_{q:.2f}.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['delta_grid'], DELTA_GRID)
            np.testing.assert_array_equal(saved['methods'], METHODS)
            assert float(saved['q']) == q
            regrets = saved['regret']
            assert regrets.shape == (B, len(DELTA_GRID), len(METHODS))
            assert np.isfinite(regrets).all()
            for di, delta in enumerate(DELTA_GRID):
                bounds = regret_bounds(q, float(delta))
                for mi, method in enumerate(METHODS):
                    values = regrets[:, di, mi]
                    mean = float(values.mean())
                    mcse = float(values.std(ddof=1)/np.sqrt(B))
                    rows.append(dict(q=q, delta1=float(delta), method=method,
                                     mean_regret=mean, mcse=mcse,
                                     ci_low=mean-CI_CRITICAL*mcse, ci_high=mean+CI_CRITICAL*mcse,
                                     **bounds[method]))
    return pd.DataFrame(rows)

#endregion
