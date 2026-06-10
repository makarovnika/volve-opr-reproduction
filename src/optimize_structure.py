"""
Stage-1 structural hyperparameter optimization with PSO (paper Fig. 7, Table 2).

A PSO swarm searches the DL architecture space (layer count, node counts,
dropout) to minimize validation RMSE.  Each particle evaluation trains a short
backprop model.  Returns the best hyperparameters and the convergence history.

This is the expensive "full optimizer" path; the default pipeline instead uses
the already-reported Table-2 optima.  Budgets are kept small so it finishes in
a few minutes on CPU.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from . import metrics as M
from .data import build_dataset
from .models import LSTMRegressor, CNNRegressor, set_seed
from .optimizers import pso


def _decode(kind: str, vec: np.ndarray) -> dict:
    """Map a continuous PSO vector to an integer hyperparameter dict."""
    if kind == "LSTM":
        n1, n2, n3, drop = vec
        return {"hidden_layers": 3,
                "nodes": [int(round(n1)), int(round(n2)), int(round(n3))],
                "dropout": float(np.clip(drop, 0.0, 0.6)),
                "recurrent_lr": [1.0, 1.0, 1.0], "solver": "sgdm",
                "gate_activation": "sigmoid", "min_batch_size": 229}
    else:  # CNN
        n1, n2, n3, drop = vec
        return {"conv_layers": 3,
                "nodes": [int(round(n1)), int(round(n2)), int(round(n3))],
                "kernel_size": [1, 1, 1],
                "dropout": [float(np.clip(drop, 0.0, 0.3))] * 3,
                "dense_layers": 3, "dense_nodes": 33, "solver": "rmsprop"}


def _quick_fit_rmse(kind, hp, ds, epochs=25):
    set_seed(C.SEED)
    if kind == "LSTM":
        model = LSTMRegressor(ds.X_train.shape[-1], hp)
    else:
        model = CNNRegressor(ds.X_train.shape[-1], hp, seq_len=ds.X_train.shape[1])
    # Random (not tail) validation split — the time-ordered tail is a
    # distribution-shifted regime that gives an unstable structure-search signal
    # (same reasoning as nsga2._quick_lstm_rmse).
    n = len(ds.X_train); n_val = max(1, int(n * 0.2))
    perm = np.random.default_rng(C.SEED).permutation(n)
    va, tr = perm[:n_val], perm[n_val:]
    Xtr = torch.tensor(ds.X_train[tr]); ytr = torch.tensor(ds.y_train[tr])
    Xva = torch.tensor(ds.X_train[va]); yva = ds.y_train[va]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss_fn(model(Xtr), ytr).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pv = ds.y_scaler_train.inverse(model(Xva).numpy())
    return M.rmse(ds.y_scaler_train.inverse(yva), pv)


def optimize_structure(kind: str, n_particles=10, n_iter=40, epochs=20):
    """Run PSO Stage-1 for 'LSTM' or 'CNN'.  Returns (best_hp, best_rmse, history)."""
    ds = build_dataset()
    bounds = np.array([[8, 32], [8, 32], [8, 32], [0.0, 0.5]])  # nodes x3, dropout

    def objective(vec):
        return _quick_fit_rmse(kind, _decode(kind, vec), ds, epochs)

    best, best_f, hist = pso(objective, bounds, n_particles=n_particles,
                             n_iter=n_iter, rng=np.random.default_rng(C.SEED))
    return _decode(kind, best), best_f, hist
