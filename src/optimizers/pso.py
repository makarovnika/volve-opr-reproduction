"""
Particle Swarm Optimization (Kennedy & Eberhart; paper ref. [33]).

Generic minimizer used for both optimization stages:
  * Stage 1 - tuning the DL structural hyperparameters (low-dim, wide bounds);
  * Stage 2 - fine-tuning the DL weights and biases (high-dim, init near a
    backprop solution).

Returns the best vector found, its fitness, and the per-iteration best-fitness
history (used to draw the convergence curves of Figs 7 & 8).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def pso(
    objective: Callable[[np.ndarray], float],
    bounds: np.ndarray,                 # (dim, 2): [low, high] per dimension
    n_particles: int = 20,
    n_iter: int = 40,
    w: float = 0.7,                     # inertia
    c1: float = 1.5,                    # cognitive
    c2: float = 1.5,                    # social
    init: np.ndarray | None = None,     # optional warm-start centre
    init_scale: float = 0.05,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
):
    rng = rng or np.random.default_rng(0)
    bounds = np.asarray(bounds, dtype=float)
    dim = bounds.shape[0]
    low, high = bounds[:, 0], bounds[:, 1]
    span = high - low

    if init is not None:
        # Warm start: cloud around the provided solution (Stage-2 weight tuning).
        X = init[None, :] + rng.normal(0, init_scale, size=(n_particles, dim)) * span
    else:
        X = low + rng.random((n_particles, dim)) * span
    X = np.clip(X, low, high)
    V = rng.normal(0, 0.1, size=(n_particles, dim)) * span

    pbest = X.copy()
    pbest_f = np.array([objective(x) for x in X])
    g = int(np.argmin(pbest_f))
    gbest, gbest_f = pbest[g].copy(), float(pbest_f[g])

    history = [gbest_f]
    for it in range(n_iter):
        r1, r2 = rng.random((n_particles, dim)), rng.random((n_particles, dim))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = np.clip(X + V, low, high)
        f = np.array([objective(x) for x in X])
        improved = f < pbest_f
        pbest[improved], pbest_f[improved] = X[improved], f[improved]
        g = int(np.argmin(pbest_f))
        if pbest_f[g] < gbest_f:
            gbest, gbest_f = pbest[g].copy(), float(pbest_f[g])
        history.append(gbest_f)
        if verbose:
            print(f"  PSO it {it+1:3d}/{n_iter}  best={gbest_f:.5f}")
    return gbest, gbest_f, np.array(history)
