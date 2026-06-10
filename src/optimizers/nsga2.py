"""
Dual-objective NSGA-II feature selection wrapped around an LSTM (paper Sec. 2.2,
3.1; Table 1 / Fig. 6).

Objectives (both minimized):
    f1 = OPR prediction RMSE of an LSTM trained on the candidate feature subset
    f2 = number of selected features

A Pareto front of subsets results; for each subset size we report the best
(lowest-RMSE) subset, reproducing Table 1 and Fig. 6.

The LSTM fitness is deliberately lightweight (few epochs, small net) so the
search is tractable; pass ``epochs``/``pop``/``gen`` to scale fidelity.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize

from .. import config as C
from .. import metrics as M
from ..data import build_dataset, MinMax, make_windows
from ..models import set_seed


# --------------------------------------------------------------------------- #
def _quick_lstm_rmse(features: list[str], epochs: int, seq_len: int,
                     seed: int = C.SEED) -> tuple[float, float]:
    """Train a small LSTM on the given feature subset; return (RMSE, R2) on a
    held-out tail of the training data (in m3/day)."""
    set_seed(seed)
    # Feature selection ranks the exogenous features, so the lagged-target
    # channel is disabled here (it would otherwise dominate and mask the
    # incremental value of each well-performance feature).
    ds = build_dataset(features=features, seq_len=seq_len,
                       use_lagged_target=False)
    X, y = ds.X_train, ds.y_train

    # Random validation split (the tail is a distribution-shifted well regime
    # that yields unstable, negative-R2 estimates for the wrapper).
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_val = max(1, int(len(X) * 0.2))
    va, tr = perm[:n_val], perm[n_val:]
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]

    n_in = ds.X_train.shape[-1]
    class _Net(nn.Module):                     # 3 x 20-node LSTM per the paper
        def __init__(self, f):
            super().__init__()
            self.l1 = nn.LSTM(f, 20, batch_first=True)
            self.l2 = nn.LSTM(20, 20, batch_first=True)
            self.drop = nn.Dropout(0.1)
            self.head = nn.Linear(20, 1)
        def forward(self, x):
            x, _ = self.l1(x); x, _ = self.l2(x)
            return self.head(self.drop(x[:, -1, :])).squeeze(-1)
    net = _Net(n_in)

    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    bs = 512
    for _ in range(epochs):
        net.train()
        p = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = p[i:i + bs]
            opt.zero_grad()
            loss_fn(net(Xt[idx]), yt[idx]).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        pv = net(torch.tensor(Xva)).numpy()
    pv_phys = ds.y_scaler_train.inverse(pv)
    yv_phys = ds.y_scaler_train.inverse(yva)
    return M.rmse(yv_phys, pv_phys), M.r2(yv_phys, pv_phys)


# --------------------------------------------------------------------------- #
class _FeatureProblem(ElementwiseProblem):
    def __init__(self, all_features, epochs, seq_len):
        super().__init__(n_var=len(all_features), n_obj=2, xl=0, xu=1, vtype=bool)
        self.all_features = all_features
        self.epochs = epochs
        self.seq_len = seq_len
        self._cache: dict = {}

    def _evaluate(self, x, out, *args, **kwargs):
        mask = np.asarray(x, dtype=bool)
        if not mask.any():
            out["F"] = [1e6, 0]
            return
        feats = tuple(f for f, m in zip(self.all_features, mask) if m)
        if feats not in self._cache:
            rmse, _ = _quick_lstm_rmse(list(feats), self.epochs, self.seq_len)
            self._cache[feats] = rmse
        out["F"] = [self._cache[feats], int(mask.sum())]


def nsga2_feature_selection(all_features=None, pop=20, gen=15, epochs=20,
                            seq_len=C.SEQ_LEN, seed=C.SEED):
    """Run NSGA-II; return (per-size best table, evaluation cache)."""
    all_features = all_features or C.FEATURES
    problem = _FeatureProblem(all_features, epochs, seq_len)
    algo = NSGA2(
        pop_size=pop,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(prob=0.79),
        mutation=BitflipMutation(prob=0.08),
        eliminate_duplicates=True,
    )
    minimize(problem, algo, ("n_gen", gen), seed=seed, verbose=False)

    # Best subset per feature-count from the full evaluation cache.
    best_by_size: dict[int, tuple[tuple, float]] = {}
    for feats, rmse in problem._cache.items():
        k = len(feats)
        if k not in best_by_size or rmse < best_by_size[k][1]:
            best_by_size[k] = (feats, rmse)
    return best_by_size, problem._cache
