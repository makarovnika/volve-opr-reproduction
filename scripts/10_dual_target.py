"""
Step 10 - Dual-target honesty reporting (Task 6).

Every headline RMSE in this project is computed on the **denoised** OPR. Because
the one-step models are dominated by OPR autocorrelation, the denoising strength
silently sets the achievable error (autocorr → 1 ⇒ RMSE → 0). To keep this
transparent, this step re-scores the saved test predictions against BOTH:

  * the **denoised** target (the SD-file OPR the models were trained on), and
  * the **raw** (pre-denoise) OPR for the same test dates
    (`raw_pipeline.compile_dataset(denoise=False)` — the date-aligned aggregation).

It also reports each target's lag-1 autocorrelation and persistence floor, so the
smoothing level is explicit.

Output: results/tables/table4b_dual_target.csv / .md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import config as C
from src import metrics as M
from src import io_utils as IO
from src import raw_pipeline as RP


def _autocorr(y):
    y = np.asarray(y, float)
    return float(np.corrcoef(y[1:], y[:-1])[0, 1])


def main():
    meas_tr, meas_te, preds_tr, preds_te = IO.load_predictions()  # denoised (SD)

    # Raw (undenoised) aggregated OPR for the same train/test date windows.
    tr_raw, te_raw = RP.compile_dataset(denoise=False)
    raw_te = te_raw["OPR"].to_numpy(float)
    n = min(len(raw_te), len(meas_te))
    raw_te, meas_te = raw_te[:n], meas_te[:n]

    print("Target smoothness (test):")
    print(f"  denoised (SD): autocorr={_autocorr(meas_te):.4f}  "
          f"persistence RMSE={M.rmse(meas_te[1:], meas_te[:-1]):.3f}")
    print(f"  raw           : autocorr={_autocorr(raw_te):.4f}  "
          f"persistence RMSE={M.rmse(raw_te[1:], raw_te[:-1]):.3f}")

    rows = []
    for m in C.MODELS:
        p = np.asarray(preds_te[m], float)[:n]
        for target, y in [("denoised", meas_te), ("raw", raw_te)]:
            d = M.all_metrics(y, p)
            rows.append({"model": m, "target": target,
                         "RMSE": round(d["RMSE"], 3), "MAE": round(d["MAE"], 3),
                         "R2": round(d["R2"], 4)})
    tab = pd.DataFrame(rows).set_index(["model", "target"])
    IO.save_table(tab.reset_index().set_index("model"), "table4b_dual_target")
    print("\nTest metrics on BOTH targets:")
    print(tab.to_string())
    print("\nNote: the headline (denoised) RMSE reflects the denoised target's "
          "smoothness (autocorr→1); the raw-OPR RMSE is the honest noisy-signal "
          "error and is several× larger.")


if __name__ == "__main__":
    main()
