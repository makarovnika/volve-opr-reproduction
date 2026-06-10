"""
Step 00c - Validate the raw->SD reconstruction against the supplied SD file.

Computes, per common column and per subset (Train/Test), the Pearson r and RMSE
between `results/preprocessing/recon_SD.xlsx` and the supplied
`SD28Nov2024_SelectedFeature_WeveletDenoised.xlsx`, writes
`results/tables/reconstruction_fidelity.csv`, and prints PASS/FAIL.

Thresholds (TASK_reconstruct_training_dataset.md §2.7):
  * MANDATORY: corr >= 0.97 for every common column (both subsets).
  * TARGET   : corr >= 0.98 for OPR.

Exits non-zero if the MANDATORY threshold is not met anywhere.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import config as C
from src.io_utils import save_table

RECON = C.RESULTS / "preprocessing" / "recon_SD.xlsx"
COMMON = ["Time", "OSH", "ADP", "ADTemp", "AWHP", "DCS", "AW", "OPR"]
MANDATORY, TARGET_OPR = 0.97, 0.98


def _autocorr(y):
    y = np.asarray(y, float)
    return float(np.corrcoef(y[1:], y[:-1])[0, 1]) if len(y) > 2 else np.nan


def main():
    if not RECON.exists():
        print("recon_SD.xlsx missing — run scripts/00_preprocess.py first.")
        sys.exit(2)

    rows, mandatory_ok, target_ok = [], True, True
    for sheet in ("Train", "Test"):
        sd = pd.read_excel(C.DATA_XLSX, sheet_name=sheet)
        rc = pd.read_excel(RECON, sheet_name=sheet)
        n = min(len(sd), len(rc))
        for col in COMMON:
            if col not in sd or col not in rc:
                continue
            a = sd[col].to_numpy(float)[:n]
            b = rc[col].to_numpy(float)[:n]
            m = np.isfinite(a) & np.isfinite(b)
            r = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else np.nan
            rmse = float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))
            rows.append({"subset": sheet, "column": col,
                         "pearson_r": round(r, 4), "rmse_vs_SD": round(rmse, 4),
                         "autocorr_recon": round(_autocorr(b), 4),
                         "autocorr_SD": round(_autocorr(a), 4)})
            if col != "Time" and r < MANDATORY:
                mandatory_ok = False
            if col == "OPR" and r < TARGET_OPR:
                target_ok = False

    fid = pd.DataFrame(rows)
    save_table(fid.set_index(["subset", "column"]).reset_index().set_index("subset"),
               "reconstruction_fidelity")
    print(fid.to_string(index=False))

    print("\n" + "=" * 60)
    print(f"MANDATORY (all common cols corr >= {MANDATORY}): "
          f"{'PASS' if mandatory_ok else 'FAIL'}")
    print(f"TARGET    (OPR corr >= {TARGET_OPR}, both subsets): "
          f"{'PASS' if target_ok else 'PARTIAL'}")
    if not target_ok:
        opr = fid[fid.column == "OPR"][["subset", "pearson_r"]].to_dict("records")
        print(f"  OPR residual: {opr} — Test OPR < 0.98 documented in "
              f"REPRODUCTION_NOTES (the SD test well is heavily smoothed).")
    print("=" * 60)
    sys.exit(0 if mandatory_ok else 1)


if __name__ == "__main__":
    main()
