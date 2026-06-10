"""
Cuckoo Optimization Algorithm / Cuckoo Search (Yang & Deb; paper ref. [34]).

Implements Levy-flight global search with fraction-pa abandonment of the worst
nests.  Same minimizer interface as ``pso`` so it is a drop-in alternative for
Stage-2 weight/bias fine-tuning.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np


def _levy(dim: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Mantegna's algorithm for Levy-distributed step lengths."""
    sigma_u = (
        math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = rng.normal(0, sigma_u, size=dim)
    v = rng.normal(0, 1, size=dim)
    return u / np.abs(v) ** (1 / beta)


def coa(
    objective: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_nests: int = 20,
    n_iter: int = 40,
    pa: float = 0.25,                  # discovery (abandonment) probability
    alpha: float = 0.01,               # step-size scaling
    beta: float = 1.5,                 # Levy exponent
    init: np.ndarray | None = None,
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
        nests = init[None, :] + rng.normal(0, init_scale, size=(n_nests, dim)) * span
    else:
        nests = low + rng.random((n_nests, dim)) * span
    nests = np.clip(nests, low, high)
    fit = np.array([objective(x) for x in nests])

    best_i = int(np.argmin(fit))
    best, best_f = nests[best_i].copy(), float(fit[best_i])
    history = [best_f]

    for it in range(n_iter):
        # --- Levy-flight new solutions ---
        # Canonical Cuckoo Search move: x_i + alpha * Levy(beta) (x_i - best).
        # (No extra Gaussian factor — that would re-randomize the already
        #  Levy-distributed step, cancel the directional pull toward `best`, and
        #  defeat the heavy-tailed exploration the algorithm is named for.)
        for i in range(n_nests):
            step = alpha * _levy(dim, beta, rng) * (nests[i] - best) * span
            cand = np.clip(nests[i] + step, low, high)
            fc = objective(cand)
            if fc < fit[i]:
                nests[i], fit[i] = cand, fc

        # --- abandon a fraction pa of nests, build new ones ---
        n_abandon = int(pa * n_nests)
        if n_abandon > 0:
            worst = np.argsort(fit)[-n_abandon:]
            for i in worst:
                d1, d2 = rng.integers(0, n_nests, 2)
                cand = np.clip(
                    nests[i] + rng.random() * (nests[d1] - nests[d2]), low, high
                )
                fc = objective(cand)
                if fc < fit[i]:
                    nests[i], fit[i] = cand, fc

        best_i = int(np.argmin(fit))
        if fit[best_i] < best_f:
            best, best_f = nests[best_i].copy(), float(fit[best_i])
        history.append(best_f)
        if verbose:
            print(f"  COA it {it+1:3d}/{n_iter}  best={best_f:.5f}")
    return best, best_f, np.array(history)
