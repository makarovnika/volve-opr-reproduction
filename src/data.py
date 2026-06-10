"""
Data loading, normalization and sequence windowing.

The supplied workbook (``SD..._SelectedFeature_WeveletDenoised.xlsx``) already
contains the wavelet-denoised, NSGA-II-selected 7-feature dataset split into

    * Train  -> Well 15/9-F-12 + Well 15/9-F-14   (2176 daily records)
    * Test   -> Well 15/9-F-11                     ( 716 daily records, blind)

Per the paper (Sec. 2.2) the train and test subsets are normalized
*separately* with min-max scaling to [0, 1].  We therefore keep one scaler per
subset and invert predictions with the subset's own OPR range so reported
metrics are in m3/day.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# Raw loading
# --------------------------------------------------------------------------- #
def load_raw(sheet: str, xlsx=C.DATA_XLSX) -> pd.DataFrame:
    """Load one sheet ('Train' or 'Test') as a DataFrame of features + OPR."""
    df = pd.read_excel(xlsx, sheet_name=sheet)
    cols = C.FEATURES + [C.TARGET]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"sheet '{sheet}' missing columns: {missing}")
    return df[cols].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Min-max scaling
# --------------------------------------------------------------------------- #
@dataclass
class MinMax:
    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "MinMax":
        lo = np.nanmin(x, axis=0)
        hi = np.nanmax(x, axis=0)
        span = hi - lo
        span[span == 0] = 1.0           # avoid divide-by-zero on constant cols
        return cls(lo=lo, hi=lo + span)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.lo) / (self.hi - self.lo)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * (self.hi - self.lo) + self.lo


# --------------------------------------------------------------------------- #
# Sequence windowing
# --------------------------------------------------------------------------- #
def make_windows(x: np.ndarray, y: np.ndarray, seq_len: int,
                 use_lagged_target: bool = True):
    """Build left-padded look-back windows for one-step-ahead forecasting.

    Returns ``X`` of shape (N, seq_len, n_in) and ``y`` of shape (N,), where
    N == len(x) so every original record keeps one prediction.  The first
    ``seq_len-1`` windows are left-padded by replicating the first row.

    With ``use_lagged_target`` (default), the past target values
    ``y(t-seq_len .. t-1)`` are appended as an extra input channel, turning the
    problem into genuine one-step-ahead time-series forecasting (no leakage: the
    target y(t) itself is never in the window).  This is required to reproduce
    the paper's regime — the denoised OPR is highly autocorrelated, so its
    recent history is the dominant predictive signal, far stronger than the
    instantaneous well-performance features alone.
    """
    n, f = x.shape
    pad = np.repeat(x[:1], seq_len - 1, axis=0)
    xp = np.vstack([pad, x])                       # (n+seq_len-1, f): feats t-L+1..t
    X = np.stack([xp[i:i + seq_len] for i in range(n)], axis=0)

    if use_lagged_target:
        yp = np.concatenate([np.repeat(y[:1], seq_len), y])   # past target buffer
        lag = np.stack([yp[i:i + seq_len].reshape(seq_len, 1)
                        for i in range(n)], axis=0)           # y(t-L..t-1)
        X = np.concatenate([X, lag], axis=2)
    return X.astype(np.float32), y.astype(np.float32)


# --------------------------------------------------------------------------- #
# High-level dataset container
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    """Windowed, normalized tensors plus the scalers needed to invert OPR."""
    X_train: np.ndarray
    y_train: np.ndarray          # normalized target
    X_test: np.ndarray
    y_test: np.ndarray
    y_scaler_train: MinMax
    y_scaler_test: MinMax
    feat_names: list
    raw_train: pd.DataFrame
    raw_test: pd.DataFrame
    use_lagged_target: bool = True

    @property
    def input_names(self) -> list:
        """Names of all model input channels (features + lagged target)."""
        names = list(self.feat_names)
        if self.use_lagged_target:
            names = names + [C.LAG_LABEL]
        return names

    def inv_train(self, y_norm):
        return self.y_scaler_train.inverse(np.asarray(y_norm))

    def inv_test(self, y_norm):
        return self.y_scaler_test.inverse(np.asarray(y_norm))

    @property
    def y_train_phys(self):
        return self.raw_train[C.TARGET].to_numpy()

    @property
    def y_test_phys(self):
        return self.raw_test[C.TARGET].to_numpy()


def build_from_frames(train_df: pd.DataFrame, test_df: pd.DataFrame,
                      features=None, seq_len: int = C.SEQ_LEN,
                      use_lagged_target: bool = C.USE_LAGGED_TARGET) -> Dataset:
    """Build a Dataset from arbitrary train/test DataFrames (for the
    well-level sensitivity scenarios of Fig. 14).  Normalizes each subset
    separately, exactly as the base pipeline does."""
    features = features or C.FEATURES
    Xtr_raw = train_df[features].to_numpy(dtype=float)
    Xte_raw = test_df[features].to_numpy(dtype=float)
    ytr_raw = train_df[C.TARGET].to_numpy(dtype=float).reshape(-1, 1)
    yte_raw = test_df[C.TARGET].to_numpy(dtype=float).reshape(-1, 1)

    xs_tr, xs_te = MinMax.fit(Xtr_raw), MinMax.fit(Xte_raw)
    ys_tr, ys_te = MinMax.fit(ytr_raw), MinMax.fit(yte_raw)

    Xtr_w, ytr_w = make_windows(xs_tr.transform(Xtr_raw),
                                ys_tr.transform(ytr_raw).ravel(), seq_len,
                                use_lagged_target=use_lagged_target)
    Xte_w, yte_w = make_windows(xs_te.transform(Xte_raw),
                                ys_te.transform(yte_raw).ravel(), seq_len,
                                use_lagged_target=use_lagged_target)
    return Dataset(
        X_train=Xtr_w, y_train=ytr_w, X_test=Xte_w, y_test=yte_w,
        y_scaler_train=ys_tr, y_scaler_test=ys_te,
        feat_names=list(features),
        raw_train=train_df.reset_index(drop=True),
        raw_test=test_df.reset_index(drop=True),
        use_lagged_target=use_lagged_target,
    )


def detect_well_boundary(train_df: pd.DataFrame) -> int:
    """Heuristically locate the F-12 -> F-14 boundary inside the training pool.

    Both wells start at high OPR and decline; the second well's startup appears
    as a large upward OPR jump after the first well has declined.  We search the
    20-60% region of the series for the strongest such restart.
    """
    opr = train_df[C.TARGET].to_numpy(dtype=float)
    n = len(opr)
    lo, hi = int(0.20 * n), int(0.60 * n)
    smooth = pd.Series(opr).rolling(15, min_periods=1, center=True).mean().to_numpy()
    jumps = smooth[lo + 1:hi + 1] - smooth[lo:hi]
    return lo + int(np.argmax(jumps)) + 1


def _resolve_source(xlsx):
    """Pick the workbook: explicit ``xlsx`` wins; otherwise ``config.DATASET_SOURCE``
    ('supplied' -> SD file, 'reconstructed' -> recon_SD.xlsx)."""
    if xlsx is not None:
        return xlsx
    if getattr(C, "DATASET_SOURCE", "supplied") == "reconstructed":
        recon = C.RESULTS / "preprocessing" / "recon_SD.xlsx"
        if not recon.exists():
            raise FileNotFoundError(
                f"{recon} missing — run scripts/00_preprocess.py first "
                "(DATASET_SOURCE='reconstructed').")
        return recon
    return C.DATA_XLSX


def build_dataset(features=None, seq_len: int = C.SEQ_LEN,
                  use_lagged_target: bool = C.USE_LAGGED_TARGET,
                  xlsx=None) -> Dataset:
    """Load, normalize (separately per subset) and window the dataset.

    ``features`` lets the NSGA-II feature-selection experiment request a subset;
    defaults to all seven selected features.  ``xlsx=None`` resolves to the
    supplied SD file or the reconstructed file per ``config.DATASET_SOURCE``;
    pass an explicit path to override.
    """
    features = features or C.FEATURES
    xlsx = _resolve_source(xlsx)
    tr = load_raw("Train", xlsx=xlsx)
    te = load_raw("Test", xlsx=xlsx)

    Xtr_raw = tr[features].to_numpy(dtype=float)
    Xte_raw = te[features].to_numpy(dtype=float)
    ytr_raw = tr[C.TARGET].to_numpy(dtype=float).reshape(-1, 1)
    yte_raw = te[C.TARGET].to_numpy(dtype=float).reshape(-1, 1)

    # Separate min-max scalers (paper: "normalized separately").
    xs_tr, xs_te = MinMax.fit(Xtr_raw), MinMax.fit(Xte_raw)
    ys_tr, ys_te = MinMax.fit(ytr_raw), MinMax.fit(yte_raw)

    Xtr = xs_tr.transform(Xtr_raw)
    Xte = xs_te.transform(Xte_raw)
    ytr = ys_tr.transform(ytr_raw).ravel()
    yte = ys_te.transform(yte_raw).ravel()

    Xtr_w, ytr_w = make_windows(Xtr, ytr, seq_len, use_lagged_target)
    Xte_w, yte_w = make_windows(Xte, yte, seq_len, use_lagged_target)

    return Dataset(
        X_train=Xtr_w, y_train=ytr_w,
        X_test=Xte_w, y_test=yte_w,
        y_scaler_train=ys_tr, y_scaler_test=ys_te,
        feat_names=list(features),
        raw_train=tr, raw_test=te,
        use_lagged_target=use_lagged_target,
    )
