"""
Figure generation — reproduces the paper's figures from computed results.

Each function saves a PNG into ``results/figures`` and returns the path.
Figure numbers refer to Makarov et al., Fuel 406 (2026) 136847.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config as C

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 10})


def _bestfit(ax, x, y):
    """Draw Y=X (black solid) and least-squares best-fit (red dashed)."""
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    ax.plot([lo, hi], [lo, hi], "k-", lw=1.3, label="Y = X")
    m, b = np.polyfit(x, y, 1)
    xs = np.array([lo, hi])
    ax.plot(xs, m * xs + b, "r--", lw=1.3, label="best fit")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", "box")


# --------------------------------------------------------------------------- #
# Fig. 4 - Pearson correlation heatmap
# --------------------------------------------------------------------------- #
def fig_correlation(df: pd.DataFrame, fname="fig04_correlation_heatmap.png"):
    cols = C.FEATURES + [C.TARGET]
    corr = df[cols].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Pearson correlation — three-well dataset (Fig. 4)")
    p = C.FIG_DIR / fname
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Fig. 2 - train/test split sensitivity
# --------------------------------------------------------------------------- #
def fig_split_sensitivity(ratios, rmses, fname="fig02_split_sensitivity.png"):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot([f"{round(r*100)}:{round((1-r)*100)}" for r in ratios], rmses,
            "o-", color="tab:blue")
    ax.set_xlabel("Train : Test split ratio")
    ax.set_ylabel("Test RMSE (m³/day)")
    ax.set_title("LSTM test RMSE vs. split ratio (Fig. 2)")
    p = C.FIG_DIR / fname
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Fig. 5 - measured vs denoised OPR
# --------------------------------------------------------------------------- #
def fig_denoise(train_noisy, train_denoised, test_noisy, test_denoised,
                fname="fig05_wavelet_denoise.png"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    for ax, noisy, den, title in [
        (axes[0], train_noisy, train_denoised, "Training subset"),
        (axes[1], test_noisy, test_denoised, "Testing subset"),
    ]:
        ax.plot(noisy, color="0.7", lw=0.8, label="measured (noisy)")
        ax.plot(den, color="tab:red", lw=1.0, label="denoised")
        ax.set_title(title); ax.set_ylabel("OPR (m³/day)"); ax.legend(loc="upper right")
    axes[-1].set_xlabel("Record index")
    fig.suptitle("Wavelet denoising of OPR (Fig. 5)")
    p = C.FIG_DIR / fname
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Fig. 6 - NSGA-II feature-count vs performance
# --------------------------------------------------------------------------- #
def fig_feature_selection(n_feats, rmse, r2, fname="fig06_feature_selection.png"):
    fig, ax1 = plt.subplots(figsize=(6, 4.2))
    ax1.plot(n_feats, rmse, "o-", color="tab:blue", label="RMSE")
    ax1.set_xlabel("Number of input features")
    ax1.set_ylabel("RMSE (m³/day)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(n_feats, r2, "s--", color="tab:red", label="R²")
    ax2.set_ylabel("R²", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.axvline(7, color="grey", ls=":", lw=1)
    ax1.set_title("NSGA-II–LSTM feature selection (Fig. 6)")
    p = C.FIG_DIR / fname
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Figs. 7 & 8 - optimizer convergence
# --------------------------------------------------------------------------- #
def fig_convergence(histories: dict, title, fname):
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, h in histories.items():
        ax.plot(np.arange(1, len(h) + 1), h, label=label)
    ax.set_xlabel("Iteration"); ax.set_ylabel("RMSE (normalized)")
    ax.set_title(title); ax.legend()
    p = C.FIG_DIR / fname
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Figs. 9 & 10 - cross plots (6 models)
# --------------------------------------------------------------------------- #
def fig_crossplots(measured, preds: dict, subset_label, fname):
    order = C.MODELS
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    letters = "abcdef"
    for ax, name, L in zip(axes.ravel(), order, letters):
        y = preds[name]
        ax.scatter(measured, y, s=6, alpha=0.4, color="tab:blue")
        _bestfit(ax, measured, y)
        ax.set_title(f"({L}) {name}")
        ax.set_xlabel("Measured OPR (m³/day)")
        ax.set_ylabel("Predicted OPR (m³/day)")
    fig.suptitle(f"Measured vs predicted OPR — {subset_label}", y=1.0)
    p = C.FIG_DIR / fname
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Fig. 11 - radar plot of total scores
# --------------------------------------------------------------------------- #
def fig_radar(totals: pd.Series, fname="fig11_radar_scores.png"):
    labels = list(totals.index)
    vals = totals.values.astype(float)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    vals_c = np.concatenate([vals, vals[:1]])
    angles_c = np.concatenate([angles, angles[:1]])
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles_c, vals_c, "o-", color="tab:purple")
    ax.fill(angles_c, vals_c, alpha=0.25, color="tab:purple")
    ax.set_xticks(angles); ax.set_xticklabels(labels)
    ax.set_title("Total performance scores (Fig. 11)")
    p = C.FIG_DIR / fname
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Fig. 12 - time-series measured vs predicted (LSTM-COA)
# --------------------------------------------------------------------------- #
def fig_timeseries(meas_train, pred_train, meas_test, pred_test,
                   fname="fig12_timeseries_lstm_coa.png"):
    n_tr = len(meas_train)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    t_tr = np.arange(n_tr)
    t_te = np.arange(n_tr, n_tr + len(meas_test))
    ax.plot(t_tr, meas_train, color="0.6", lw=0.8, label="measured (train)")
    ax.plot(t_tr, pred_train, color="tab:blue", lw=0.9, label="predicted (train)")
    ax.plot(t_te, meas_test, color="0.2", lw=0.8, label="measured (test)")
    ax.plot(t_te, pred_test, color="tab:red", lw=0.9, label="predicted (test)")
    ax.axvline(n_tr, color="k", ls="--", lw=1)
    ax.set_xlabel("Record index (time order)")
    ax.set_ylabel("OPR (m³/day)")
    ax.set_title("LSTM-COA: measured vs predicted OPR (Fig. 12)")
    ax.legend(ncol=2, fontsize=8)
    p = C.FIG_DIR / fname
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Fig. 14 - sensitivity-scenario cross plots
# --------------------------------------------------------------------------- #
def fig_sensitivity(scenarios: dict, fname="fig14_sensitivity.png"):
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(10, 4.5 * len(scenarios)))
    if len(scenarios) == 1:
        axes = axes[None, :]
    for row, (sc_name, sc) in enumerate(scenarios.items()):
        for col, part in enumerate(["train", "test"]):
            ax = axes[row, col]
            meas, pred, mae = sc[part]
            ax.scatter(meas, pred, s=6, alpha=0.4, color="tab:green")
            _bestfit(ax, meas, pred)
            ax.set_title(f"{sc_name} — {part} (MAE={mae:.3f})")
            ax.set_xlabel("Measured OPR (m³/day)")
            ax.set_ylabel("Predicted OPR (m³/day)")
    fig.suptitle("LSTM-COA generalizability scenarios (Fig. 14)", y=1.0)
    p = C.FIG_DIR / fname
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return p
