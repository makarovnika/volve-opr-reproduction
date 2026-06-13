"""
Baseline models (TZ-01 §4).
Persistence, HRS-rule predictor, Ridge, XGBoost, LightGBM.
Runs both regimes A and B where applicable; 5 seeds for stochastic models.
"""
import os

import numpy as np
import pandas as pd

from .config import TARGETS, PREDICTIONS_DIR, EXPECTED
from .data_volve import make_windows
from .evaluation import rmse, evaluate, append_master_table, block_bootstrap_rmse, \
    persistence_rmse
from .seed import set_seed

L_GRID    = [30, 60, 90]   # used by Ridge only (full grid); GBM use ML_L_GRID
ML_L_GRID = [30]          # GBM/Ridge on 8100 flatten-feats is too slow; fix L=30
N_SEEDS   = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_predictions(name: str, regime: str, dates, y_true, y_pred) -> str:
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    fname = f"{name}_{regime}.csv"
    path  = os.path.join(PREDICTIONS_DIR, fname)
    df = pd.DataFrame({
        "DATE":      dates,
        "OPR_pred":  y_pred[:, 0],
        "GPR_pred":  y_pred[:, 1],
        "WPR_pred":  y_pred[:, 2],
        "OPR_true":  y_true[:, 0],
        "GPR_true":  y_true[:, 1],
        "WPR_true":  y_true[:, 2],
    })
    df.to_csv(path, index=False)
    return path


def _best_L(model_factory, ds, regime, l_grid=None):
    """Return best_L chosen by val RMSE."""
    if l_grid is None:
        l_grid = L_GRID
    best_L, best_val_rmse = None, np.inf
    for L in l_grid:
        ws = make_windows(ds, L, regime)
        if len(ws["X_train"]) == 0:
            continue
        X_tr = ws["X_train"].reshape(len(ws["X_train"]), -1)
        X_va = ws["X_val"].reshape(len(ws["X_val"]), -1)
        y_tr, y_va = ws["y_train"], ws["y_val"]
        val_rmse_sum = 0.0
        for i in range(3):
            m = model_factory()
            m.fit(X_tr, y_tr[:, i])
            val_rmse_sum += rmse(y_va[:, i], m.predict(X_va))
        avg_val_rmse = val_rmse_sum / 3
        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            best_L = L
    return best_L


def _train_and_score(name, model_factory, ds, regime, seed, l_grid=None):
    """Train on best-L, score on test, return rows + save files."""
    set_seed(seed)
    L = _best_L(model_factory, ds, regime, l_grid=l_grid)
    ws = make_windows(ds, L, regime)
    X_tr = ws["X_train"].reshape(len(ws["X_train"]), -1)
    X_va = ws["X_val"].reshape(len(ws["X_val"]), -1)
    X_te = ws["X_test"].reshape(len(ws["X_test"]), -1)
    y_tr, y_te = ws["y_train"], ws["y_test"]

    # Re-train on train+val for final model? TZ says "pick best on val, report L".
    # Keep train-only model for test predictions (cleaner leakage boundary).
    models = []
    for i in range(3):
        m = model_factory()
        m.fit(X_tr, y_tr[:, i])
        models.append(m)

    y_pred = np.column_stack([m.predict(X_te) for m in models])
    y_pred = np.clip(y_pred, 0.0, None)  # rates cannot be negative

    y_train_raw = ds["y_raw"][ds["train_mask"]]
    persist_pred = None
    if regime == "A":
        te_i = np.where(ds["test_mask"])[0]
        if len(te_i) == len(y_te) and te_i[0] > 0:
            persist_pred = ds["y_raw"][te_i - 1]
    rows = evaluate(y_te, y_pred, split="test", model=name, regime=regime,
                    seed=seed, y_train=y_train_raw, persist_pred=persist_pred)
    for row in rows:
        row["best_L"] = L
        lo, hi = block_bootstrap_rmse(
            y_te[:, TARGETS.index(row["target"])],
            y_pred[:, TARGETS.index(row["target"])],
        )
        row["RMSE_CI_lo"] = round(lo, 4)
        row["RMSE_CI_hi"] = round(hi, 4)

    if seed == 0:   # save predictions from the canonical (first) seed only
        _save_predictions(name, regime, ds["test_dates"], y_te, y_pred)
    print(f"  {name} | regime={regime} | seed={seed} | L={L} | "
          f"OPR R2={rows[0]['R2']:.3f}")
    return rows


# ---------------------------------------------------------------------------
# Individual baselines
# ---------------------------------------------------------------------------
def run_persistence(ds) -> list:
    """Regime A only: y(t) = y(t-1)."""
    y_full = ds["y_raw"]
    n_test = ds["counts"]["test_rows"]
    ts     = len(y_full) - n_test
    y_true = y_full[ts:]
    y_pred = y_full[ts - 1 : ts - 1 + n_test]
    y_train_raw = y_full[ds["train_mask"]]
    rows   = evaluate(y_true, y_pred, split="test", model="persistence",
                      regime="A", seed=0, y_train=y_train_raw,
                      persist_pred=y_pred)
    for r in rows:
        lo, hi = block_bootstrap_rmse(
            y_true[:, TARGETS.index(r["target"])],
            y_pred[:, TARGETS.index(r["target"])],
        )
        r["RMSE_CI_lo"] = round(lo, 4)
        r["RMSE_CI_hi"] = round(hi, 4)
    _save_predictions("persistence", "A", ds["test_dates"], y_true, y_pred)
    exp_opr = EXPECTED["test_rows"]   # just a note; actual assertion is in tests
    print(f"  persistence | regime=A | OPR R2={rows[0]['R2']:.3f}")
    return rows


def run_hrs_rule(ds) -> list:
    """
    HRS-rule predictor (both regimes, same predictions):
    if sum(HRS) = 0 for all wells → predict 0; else predict train mean.
    Reports how often the rule fires on test.
    """
    wide      = ds["wide"]
    feat_cols = ds["feat_cols"]
    # Identify HRS columns (one per well)
    hrs_cols  = [c for c in feat_cols if c.endswith("_HRS")]
    field_hrs = wide[hrs_cols].sum(axis=1)

    y_all  = ds["y_raw"]
    tr_i   = np.where(ds["train_mask"])[0]
    te_i   = np.where(ds["test_mask"])[0]
    train_mean = y_all[tr_i].mean(axis=0)   # per-target train mean

    y_test = y_all[te_i]
    hrs_te = field_hrs.values[te_i]
    y_pred = np.where(
        hrs_te[:, None] == 0.0,
        0.0,
        train_mean[None, :]
    )
    fires = int((hrs_te == 0.0).sum())
    print(f"  HRS-rule | fires on test: {fires}/{len(te_i)} days")

    y_train_raw = y_all[tr_i]
    persist_pred = y_all[te_i - 1] if te_i[0] > 0 else None
    all_rows = []
    for regime in ("A", "B"):
        rows = evaluate(y_test, y_pred, split="test", model="hrs_rule",
                        regime=regime, seed=0, y_train=y_train_raw,
                        persist_pred=(persist_pred if regime == "A" else None))
        for r in rows:
            r["hrs_rule_fires"] = fires
            lo, hi = block_bootstrap_rmse(
                y_test[:, TARGETS.index(r["target"])],
                y_pred[:, TARGETS.index(r["target"])],
            )
            r["RMSE_CI_lo"] = round(lo, 4)
            r["RMSE_CI_hi"] = round(hi, 4)
        all_rows.extend(rows)

    _save_predictions("hrs_rule", "B", ds["test_dates"], y_test, y_pred)
    return all_rows


def run_ridge(ds) -> list:
    from sklearn.linear_model import Ridge
    all_rows = []
    for regime in ("A", "B"):
        for seed in range(N_SEEDS):
            s = seed  # capture value for lambda
            rows = _train_and_score(
                "ridge", lambda s=s: Ridge(alpha=1.0, random_state=s),
                ds, regime, seed
            )
            all_rows.extend(rows)
    return all_rows


def run_xgboost(ds) -> list:
    from xgboost import XGBRegressor
    all_rows = []
    for regime in ("A", "B"):
        for seed in range(N_SEEDS):
            s = seed  # capture value for lambda
            rows = _train_and_score(
                "xgboost",
                lambda s=s: XGBRegressor(
                    n_estimators=300, learning_rate=0.05, max_depth=4,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                    tree_method="hist",
                    random_state=s, n_jobs=-1, verbosity=0,
                ),
                ds, regime, seed, l_grid=ML_L_GRID
            )
            all_rows.extend(rows)
    return all_rows


def run_lightgbm(ds) -> list:
    from lightgbm import LGBMRegressor
    all_rows = []
    for regime in ("A", "B"):
        for seed in range(N_SEEDS):
            s = seed  # capture value for lambda
            rows = _train_and_score(
                "lightgbm",
                lambda s=s: LGBMRegressor(
                    n_estimators=300, learning_rate=0.05, num_leaves=31,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=s, n_jobs=-1, verbose=-1,
                ),
                ds, regime, seed, l_grid=ML_L_GRID
            )
            all_rows.extend(rows)
    return all_rows


def run_all_baselines(ds) -> str:
    """Run all baselines and write results to the master table. Returns CSV path."""
    print("[baselines] Running persistence (Regime A) ...")
    rows = run_persistence(ds)
    print("[baselines] Running HRS-rule predictor ...")
    rows += run_hrs_rule(ds)
    print("[baselines] Running Ridge ...")
    rows += run_ridge(ds)
    print("[baselines] Running XGBoost ...")
    rows += run_xgboost(ds)
    print("[baselines] Running LightGBM ...")
    rows += run_lightgbm(ds)
    path = append_master_table(rows)
    print(f"[baselines] Master table: {path}")
    return path
