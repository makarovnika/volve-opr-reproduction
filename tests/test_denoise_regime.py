"""
Tests for the denoising regime and the no-imputation decision (Task 2.2, 2.3).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C
from src import raw_pipeline as RP


def _corr(a, b):
    n = min(len(a), len(b))
    a, b = np.asarray(a, float)[:n], np.asarray(b, float)[:n]
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def test_denoise_opr_target_train():
    """Denoised OPR matches the SD file with corr >= 0.98 on the train subset."""
    tr, _ = RP.compile_dataset(denoise=True)
    sd = pd.read_excel(C.DATA_XLSX, sheet_name="Train")["OPR"].to_numpy(float)
    assert _corr(sd, tr["OPR"].to_numpy(float)) >= 0.98


def test_imputation_worsens_downhole_corr():
    """Imputing F-12's dead downhole sensors WORSENS ADP/ADTemp corr vs SD —
    justifying the default clean=False (Task 2.3), not an arbitrary choice."""
    sd = pd.read_excel(C.DATA_XLSX, sheet_name="Train")
    tr_noimp, _ = RP.compile_dataset(denoise=True, clean=False)
    tr_imp, _ = RP.compile_dataset(denoise=True, clean=True)
    for col in ["ADP", "ADTemp"]:
        c_noimp = _corr(sd[col].to_numpy(float), tr_noimp[col].to_numpy(float))
        c_imp = _corr(sd[col].to_numpy(float), tr_imp[col].to_numpy(float))
        assert c_noimp > c_imp, f"{col}: no-impute {c_noimp:.3f} !> impute {c_imp:.3f}"


def test_denoise_params_from_config():
    """Denoising params live in config (single source of truth), not hard-coded."""
    assert set(C.DENOISE) >= {"wavelet", "level", "rule", "scale", "mode"}
    assert C.DENOISE["rule"] in {"rigrsure", "sqtwolog", "heursure", "minimaxi"}
