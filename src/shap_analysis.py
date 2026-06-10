"""
SHAP interpretability of the best model, LSTM-COA (paper Sec. 3.4, Fig. 13).

Uses a model-agnostic KernelExplainer over a feature-attribution wrapper: the
trained sequence model is queried on look-back windows, but SHAP attributes
importance to the *current-step* feature values (the interpretable units the
paper reports).  Produces the bee-swarm (13a) and mean-|SHAP| bar plot (13b).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

from . import config as C


def shap_analysis(model, ds, n_background=60, n_explain=150,
                  seed=C.SEED, fname="fig13_shap_lstm_coa.png"):
    rng = np.random.default_rng(seed)
    seq_len = ds.X_train.shape[1]
    n_in = ds.X_train.shape[-1]
    n_feat = len(ds.feat_names)                      # 7 exogenous features

    ex_idx = rng.choice(len(ds.X_train), size=min(n_explain, len(ds.X_train)),
                        replace=False)
    bg_idx = rng.choice(len(ds.X_train), size=min(n_background, len(ds.X_train)),
                        replace=False)

    # Per-explained-row template windows (keep real lag channel + history).
    templates = ds.X_train[ex_idx].copy()            # (E, seq_len, n_in)
    X_explain = ds.X_train[ex_idx][:, -1, :n_feat]    # current-step features only
    background = ds.X_train[bg_idx][:, -1, :n_feat]

    @torch.no_grad()
    def predict_features(F2d, template):
        F2d = np.asarray(F2d, dtype=np.float32)
        B = F2d.shape[0]
        win = np.repeat(template[None], B, axis=0)    # (B, seq_len, n_in)
        win[:, -1, :n_feat] = F2d                      # perturb ONLY the current step
        return model(torch.tensor(win)).cpu().numpy()

    # Explain each row against the shared background, then stack SHAP values.
    sv_rows = []
    for i in range(len(X_explain)):
        f_i = lambda Z, t=templates[i]: predict_features(Z, t)
        expl = shap.KernelExplainer(f_i, background)
        sv_rows.append(expl.shap_values(X_explain[i:i + 1], nsamples=60,
                                        silent=True))
    sv = np.asarray(sv_rows).reshape(len(X_explain), n_feat)

    feat_names = list(ds.feat_names)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]           # most important first

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) manual bee-swarm: SHAP value vs feature row, colored by feature value.
    ax = axes[0]
    rng2 = np.random.default_rng(seed)
    for yi, fi in enumerate(order[::-1]):        # least important at bottom
        vals = sv[:, fi]
        col = X_explain[:, fi]                    # normalized feature value [0,1]
        jitter = (rng2.random(len(vals)) - 0.5) * 0.6
        sc = ax.scatter(vals, np.full_like(vals, yi) + jitter, c=col,
                        cmap="coolwarm", s=10, alpha=0.7, vmin=0, vmax=1)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feat_names[i] for i in order[::-1]])
    ax.set_xlabel("SHAP value (impact on OPR prediction)")
    ax.set_title("(a) SHAP bee-swarm — LSTM-COA")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("feature value (low → high)")

    # (b) mean |SHAP| bar
    axes[1].barh([feat_names[i] for i in order][::-1],
                 mean_abs[order][::-1], color="tab:blue")
    axes[1].set_xlabel("mean(|SHAP value|)")
    axes[1].set_title("(b) Feature importance")

    p = C.FIG_DIR / fname
    fig.tight_layout(); fig.savefig(p, dpi=200); plt.close(fig)

    ranking = [(feat_names[i], float(mean_abs[i])) for i in order]
    return p, ranking
