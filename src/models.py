"""
PyTorch implementations of the two core deep-learning models.

Both are sequence regressors: input shape (batch, seq_len, n_features),
output a single OPR value (normalized) for the last timestep of the window.

* LSTMRegressor - stacked LSTM (paper Table 2: 3 hidden layers, 23/23/27 nodes,
  dropout 0.4748, sigmoid gate activation is the LSTM default in PyTorch).
* CNNRegressor  - 1-D temporal CNN (3 conv layers 25/19/26 filters, kernel 1,
  dropout 0.0729, 3 dense layers, 33 dense nodes).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import config as C


def set_seed(seed: int = C.SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# LSTM
# --------------------------------------------------------------------------- #
class _AnchorMixin:
    """Adds an autoregressive anchor on the lagged-target channel to a raw net
    output, BEFORE the (separately-normalized) inverse transform.

    The lagged-target channel is the last input feature; its last ``K`` timesteps
    are OPR(t-1 .. t-K), i.e. ``x[:, -k, -1]`` = OPR(t-k).  Two modes:

    * fixed (``anchor_learnable=False``): ``anchor_beta * OPR(t-1)`` (lag-1 only);
    * learnable (``ANCHOR_LEARNABLE``): ``Σ_k a_k * OPR(t-k)`` with learnable
      ``a_k`` (Task 2), init ``[anchor_beta, 0, …]`` so it starts identical to the
      fixed anchor and can grow into an AR(K) baseline.

    Set ``anchor_beta=0`` and learnable off to disable (features-only ablation).
    """
    anchor_beta: float = C.ANCHOR_BETA

    def _make_anchor(self, anchor_beta, learnable, lags):
        """Call from a model __init__ to set up the anchor."""
        self.anchor_beta = anchor_beta
        self.anchor_lags = int(lags)
        if learnable and anchor_beta is not None:
            init = torch.zeros(self.anchor_lags)
            init[0] = float(anchor_beta)
            self.anchor_w = nn.Parameter(init)
        else:
            self.anchor_w = None

    def _anchor(self, x, raw):
        if x.shape[-1] < 1:
            return raw
        if getattr(self, "anchor_w", None) is not None:    # learnable multi-lag
            K = min(self.anchor_lags, x.shape[1])
            lags = x[:, -K:, -1].flip(1)                    # OPR(t-1), OPR(t-2), …
            return (lags * self.anchor_w[:K]).sum(dim=1) + raw
        if self.anchor_beta:                               # fixed lag-1
            return self.anchor_beta * x[:, -1, -1] + raw
        return raw


class LSTMRegressor(_AnchorMixin, nn.Module):
    def __init__(self, n_features: int, hp: dict | None = None,
                 anchor_beta: float = C.ANCHOR_BETA):
        super().__init__()
        hp = hp or C.LSTM_HPARAMS
        nodes = hp["nodes"]
        self._make_anchor(anchor_beta, C.ANCHOR_LEARNABLE, C.ANCHOR_LAGS)
        self.layers = nn.ModuleList()
        in_size = n_features
        for h in nodes:
            self.layers.append(nn.LSTM(in_size, h, batch_first=True))
            in_size = h
        self.dropout = nn.Dropout(hp["dropout"])
        self.head = nn.Linear(in_size, 1)

    def forward(self, x):                       # x: (B, T, F)
        out = x
        for lstm in self.layers:
            out, _ = lstm(out)
        out = out[:, -1, :]                     # last timestep
        out = self.dropout(out)
        raw = self.head(out).squeeze(-1)        # (B,)
        return self._anchor(x, raw)


# --------------------------------------------------------------------------- #
# CNN
# --------------------------------------------------------------------------- #
class CNNRegressor(_AnchorMixin, nn.Module):
    def __init__(self, n_features: int, hp: dict | None = None,
                 seq_len: int = C.SEQ_LEN, anchor_beta: float = C.ANCHOR_BETA):
        super().__init__()
        hp = hp or C.CNN_HPARAMS
        self._make_anchor(anchor_beta, C.ANCHOR_LEARNABLE, C.ANCHOR_LAGS)
        filters = hp["nodes"]
        ksz = hp["kernel_size"]
        drop = hp["dropout"]
        convs = []
        in_c = n_features
        for f, k, d in zip(filters, ksz, drop):
            convs += [nn.Conv1d(in_c, f, kernel_size=k, padding=k // 2),
                      nn.ReLU(),
                      nn.Dropout(d)]
            in_c = f
        self.conv = nn.Sequential(*convs)
        # Compute the post-conv flattened dim from a dry run rather than assuming
        # length is preserved (true only for odd kernels with padding=k//2).
        # Run in eval mode so dropout does NOT sample and advance the global RNG
        # between this probe and the subsequent dense-layer initialization.
        self.conv.eval()
        with torch.no_grad():
            probe = torch.zeros(1, n_features, seq_len)
            self.flat_dim = self.conv(probe).flatten(1).shape[1]
        self.conv.train()
        dense = []
        d_in = self.flat_dim
        for _ in range(hp["dense_layers"]):
            dense += [nn.Linear(d_in, hp["dense_nodes"]), nn.ReLU()]
            d_in = hp["dense_nodes"]
        self.dense = nn.Sequential(*dense)
        self.head = nn.Linear(d_in, 1)

    def forward(self, x):                       # x: (B, T, F)
        out = x.transpose(1, 2)                 # (B, F, T) for Conv1d
        out = self.conv(out)
        out = out.flatten(1)
        out = self.dense(out)
        raw = self.head(out).squeeze(-1)
        return self._anchor(x, raw)


def build_model(kind: str, n_features: int, seq_len: int = C.SEQ_LEN) -> nn.Module:
    """Factory: 'LSTM' or 'CNN' (the optimizer suffix is handled in train.py)."""
    base = kind.split("-")[0].upper()
    if base == "LSTM":
        return LSTMRegressor(n_features)
    if base == "CNN":
        return CNNRegressor(n_features, seq_len=seq_len)
    raise ValueError(f"unknown model kind: {kind}")


# --------------------------------------------------------------------------- #
# Flat-parameter helpers (used by PSO / COA weight tuning, Stage 2)
# --------------------------------------------------------------------------- #
def get_flat_params(model: nn.Module) -> np.ndarray:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).cpu().numpy()


def set_flat_params(model: nn.Module, flat: np.ndarray):
    flat = torch.as_tensor(flat, dtype=torch.float32)
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[i:i + n].view_as(p))
            i += n
