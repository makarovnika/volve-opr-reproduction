"""
Step 00b - Reconstruct the noise-removal (wavelet denoising) process.

The supplied files contain TWO distinct denoising regimes:

  * NM..NoiseFree (Dec 2023)  -> HEAVY low-pass: only the wavelet approximation
    band kept (all detail discarded).  Lag-1 autocorrelation ~0.9997.
  * SD..._WeveletDenoised (Nov 2024, the paper's dataset) -> LIGHT SURE
    soft-threshold denoising that preserves transients.  Lag-1 autocorr ~0.905.

This script recreates both, applied to the raw OPR (= BORE_OIL_VOL/ON_STREAM_HRS)
computed from the original Volve export, and characterizes each against the
supplied targets (autocorrelation, std, variance-retained).  It saves a stepwise
comparison figure and a characterization table.

Outputs:
  results/figures/fig05b_denoising_regimes.png
  results/tables/denoising_characterization.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src.io_utils import save_table
from src.preprocessing import wavelet_denoise, wavelet_approx
from src.raw_pipeline import load_raw, build_well_frame


def _autocorr(x):
    x = np.asarray(x, float)
    return float(np.corrcoef(x[1:], x[:-1])[0, 1])


def _char(name, x):
    x = np.asarray(x, float)
    return {"series": name, "n": len(x), "std": round(float(x.std()), 2),
            "lag1_autocorr": round(_autocorr(x), 4),
            "min": round(float(x.min()), 2), "max": round(float(x.max()), 2)}


def main():
    raw = load_raw()
    # Noisy OPR for Well 15/9-F-14 (longest clean producer) as the demo signal.
    f14 = build_well_frame(raw, "15/9-F-14")
    noisy = f14["OPR"].to_numpy(float)

    soft = wavelet_denoise(noisy, wavelet="sym8")        # light (paper / SD)
    approx = wavelet_approx(noisy, wavelet="db4", level=4)  # heavy (NoiseFree)

    rows = [_char("raw OPR (F-14)", noisy),
            _char("soft-threshold (SD-style)", soft),
            _char("approx-only (NoiseFree-style)", approx)]

    # Reference targets from the supplied processed files.
    try:
        nf = pd.read_excel(C.INPUT_DIR / "NM20Dect2023NoiseFreeTrain.xlsx",
                           sheet_name="Denoised data")["value"].to_numpy()
        rows.append(_char("TARGET NoiseFree file", nf))
    except Exception:
        pass
    try:
        sd = pd.read_excel(C.DATA_XLSX, sheet_name="Train")["OPR"].to_numpy()
        rows.append(_char("TARGET SD file (paper)", sd))
    except Exception:
        pass

    tab = pd.DataFrame(rows).set_index("series")
    # variance retained vs raw
    var_raw = float(np.var(noisy))
    tab["var_retained_%"] = [round(100 * np.var(noisy) / var_raw, 1),
                             round(100 * np.var(soft) / var_raw, 1),
                             round(100 * np.var(approx) / var_raw, 1)] + \
                            [np.nan] * (len(tab) - 3)
    save_table(tab, "denoising_characterization")
    print(tab.to_string())

    # Figure: raw vs the two regimes (zoom on a spiky window to show the contrast)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    axes[0].plot(noisy, color="0.7", lw=0.7, label="raw OPR (noisy)")
    axes[0].plot(soft, color="tab:blue", lw=0.9, label="soft-threshold (light, SD/paper)")
    axes[0].plot(approx, color="tab:red", lw=1.1, label="approx-only (heavy, NoiseFree)")
    axes[0].set_title("Wavelet denoising regimes — Well 15/9-F-14 OPR")
    axes[0].set_ylabel("OPR (m³/day)"); axes[0].legend(loc="upper right", fontsize=8)

    z0, z1 = 150, 450
    axes[1].plot(range(z0, z1), noisy[z0:z1], color="0.7", lw=0.9, label="raw")
    axes[1].plot(range(z0, z1), soft[z0:z1], color="tab:blue", lw=1.1, label="soft (keeps spikes)")
    axes[1].plot(range(z0, z1), approx[z0:z1], color="tab:red", lw=1.4, label="approx (smooths spikes)")
    axes[1].set_title("Zoom: soft thresholding preserves transients; approx-only removes them")
    axes[1].set_xlabel("Record index"); axes[1].set_ylabel("OPR (m³/day)")
    axes[1].legend(loc="upper right", fontsize=8)

    p = C.FIG_DIR / "fig05b_denoising_regimes.png"
    fig.tight_layout(); fig.savefig(p, dpi=200); plt.close(fig)
    print("Saved", p.name)


if __name__ == "__main__":
    main()
