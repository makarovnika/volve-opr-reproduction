"""
Stage-1 architecture-family reconstructions (TASK §4 Case B) on the SHARED
Stage-2 pipeline.  These are minimal faithful TCN / GNN models — NOT the original
published checkpoints (that code is lost) — used to place the architecture
families on the same protocol as Stage-2.

Contract: consume make_windows() tensors (X scaled, y RAW), train on y scaled by
the dataset's y_scaler, and return RAW-unit test predictions for evaluation.py.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import PRODUCERS, INJECTORS, WELL_ABBREV

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WELLS = PRODUCERS + INJECTORS                      # 0..4 producers, 5..6 injectors


# --------------------------------------------------------------------------- #
# Node grouping for the GNN (map wide-frame columns -> per-well node features)
# --------------------------------------------------------------------------- #
def build_node_map(feat_cols: list[str], n_total_channels: int):
    """Return (node_idx list[7][k], shared_idx, node_dim).

    Each well node gets its own channels (well-prefixed + active/imputed/
    inconsistent flags); AW / f12_bhp_dead and any Regime-A lagged-target
    channels (indices >= len(feat_cols)) are SHARED and broadcast to all nodes.
    """
    node_idx = []
    claimed = set()
    for well in WELLS:
        ab = WELL_ABBREV[well]
        idx = [i for i, c in enumerate(feat_cols)
               if c.startswith(ab + "_") or c == f"active_{ab}"
               or c.startswith(f"imputed_{ab}_") or c == f"inconsistent_{ab}"]
        node_idx.append(idx)
        claimed.update(idx)
    shared = [i for i in range(len(feat_cols)) if i not in claimed]
    # Regime-A appends lagged targets after the feat_cols block -> shared.
    shared += list(range(len(feat_cols), n_total_channels))
    node_dim = max(len(ix) for ix in node_idx) + len(shared)
    return node_idx, shared, node_dim


# --------------------------------------------------------------------------- #
# TCN — dilated causal 1-D conv stack
# --------------------------------------------------------------------------- #
class _CausalBlock(nn.Module):
    def __init__(self, c_in, c_out, k, dilation, p=0.1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, k, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, k, dilation=dilation)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x):                          # x: (B, C, L)
        res = x if self.down is None else self.down(x)
        y = self.conv1(nn.functional.pad(x, (self.pad, 0)))
        y = self.drop(self.relu(y))
        y = self.conv2(nn.functional.pad(y, (self.pad, 0)))
        y = self.drop(self.relu(y))
        return self.relu(y + res)


class TCN(nn.Module):
    """3-block dilated causal TCN over the L-day window -> 3 field targets."""
    def __init__(self, n_feat, hidden=48, k=3, blocks=3, n_out=3):
        super().__init__()
        layers, c = [], n_feat
        for b in range(blocks):
            layers.append(_CausalBlock(c, hidden, k, dilation=2 ** b))
            c = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, n_out))

    def forward(self, x):                          # x: (B, L, F)
        h = self.tcn(x.transpose(1, 2))            # (B, hidden, L)
        return self.head(h[:, :, -1])              # last timestep


# --------------------------------------------------------------------------- #
# GNN — per-well temporal encoder (GRU) + graph message passing + sum readout
# --------------------------------------------------------------------------- #
def _adjacency():
    """Producer<->injector + producer<->producer spatial adjacency (+ self loops),
    symmetrically normalised.  7x7, order = PRODUCERS(0-4) + INJECTORS(5-6)."""
    n = len(WELLS); A = np.eye(n)
    prod = list(range(len(PRODUCERS))); inj = list(range(len(PRODUCERS), n))
    for i in prod:                                 # producers spatially adjacent
        for j in prod:
            A[i, j] = 1.0
    for i in prod:                                 # producer <-> injector (sweep)
        for j in inj:
            A[i, j] = A[j, i] = 1.0
    d = A.sum(1); Dm = np.diag(1.0 / np.sqrt(d))
    return torch.tensor(Dm @ A @ Dm, dtype=torch.float32)


class GNN(nn.Module):
    def __init__(self, node_idx, shared, node_dim, hidden=48, n_out=3, gnn_layers=2):
        super().__init__()
        self.node_idx = node_idx
        self.shared = shared
        self.node_dim = node_dim
        self.n_nodes = len(node_idx)
        self.register_buffer("A", _adjacency())
        self.enc = nn.GRU(node_dim, hidden, batch_first=True)   # shared across nodes
        self.gnn = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(gnn_layers))
        self.relu = nn.ReLU()
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, n_out))

    def _to_nodes(self, x):                         # x: (B, L, C) -> (B*N, L, node_dim)
        B, L, _ = x.shape
        sh = x[:, :, self.shared] if self.shared else x[:, :, :0]
        feats = []
        for idx in self.node_idx:
            own = x[:, :, idx] if idx else x[:, :, :0]
            nd = torch.cat([own, sh], dim=2)
            pad = self.node_dim - nd.shape[2]
            if pad > 0:
                nd = nn.functional.pad(nd, (0, pad))
            feats.append(nd)
        return torch.stack(feats, dim=1).reshape(B * self.n_nodes, L, self.node_dim)

    def forward(self, x):
        B = x.shape[0]
        nodes = self._to_nodes(x)
        _, h = self.enc(nodes)                       # (1, B*N, H)
        H = h.squeeze(0).reshape(B, self.n_nodes, -1)   # (B, N, H)
        for lin in self.gnn:                         # message passing
            H = self.relu(lin(torch.einsum("ij,bjh->bih", self.A, H)))
        prod = H[:, :len(PRODUCERS), :].sum(dim=1)   # field = sum over producers
        return self.head(prod)


# --------------------------------------------------------------------------- #
# Shared training / evaluation harness
# --------------------------------------------------------------------------- #
def train_predict(model: nn.Module, ws: dict, y_scaler, *, seed: int,
                  epochs=80, patience=12, lr=2e-3, batch=256):
    """Train on scaled y, early-stop on SCALED val RMSE (scale-invariant across
    the 3 very-different-scale targets), return RAW test+val predictions."""
    torch.manual_seed(seed); np.random.seed(seed)
    # seed.py (copied verbatim from Stage-2) sets cudnn.deterministic=True /
    # benchmark=False, which makes the dilated-conv path ~30x slower. Determinism
    # here comes from the seeded init + data-shuffle order; re-enable autotuning
    # so runs finish in seconds (conv outputs then differ only at ~float tol).
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    model = model.to(DEVICE)

    def t(a): return torch.tensor(a, dtype=torch.float32, device=DEVICE)
    Xtr, Xva, Xte = t(ws["X_train"]), t(ws["X_val"]), t(ws["X_test"])
    ytr = t(y_scaler.transform(ws["y_train"]))
    yva_sc = t(y_scaler.transform(ws["y_val"]))      # scaled val targets
    center = t(y_scaler.center_); scale = t(y_scaler.scale_)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    n = len(Xtr); best_rmse, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch):
            bi = perm[s:s + batch]
            opt.zero_grad()
            lossf(model(Xtr[bi]), ytr[bi]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            rmse_va = float(torch.sqrt(torch.mean((model(Xva) - yva_sc) ** 2)))
        if rmse_va < best_rmse * (1 - 1e-3):         # relative improvement
            best_rmse, best_state, bad = rmse_va, {k: v.cpu().clone()
                                                   for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pte = (model(Xte) * scale + center).cpu().numpy()
        pva = (model(Xva) * scale + center).cpu().numpy()
    return pte, pva, best_rmse
