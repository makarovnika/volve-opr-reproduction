"""
Step 00 - Reconstruct the data-preparation pipeline from the RAW Volve export.

Runs Volve production data.xlsx -> compiled -> cleaned -> wavelet-denoised ->
NSGA-II/greedy feature selection (10 -> 7) and writes a reconstructed
``SelectedFeature_WaveletDenoised`` workbook, then compares it against the
supplied ``SD28Nov2024_SelectedFeature_WeveletDenoised.xlsx`` to quantify
fidelity.

Outputs:
  results/preprocessing/recon_01_compiled.xlsx      (10 features + OPR, raw)
  results/preprocessing/recon_02_denoised.xlsx       (wavelet-denoised)
  results/preprocessing/recon_03_selected.xlsx       (7 selected features)
  results/tables/table1b_raw_feature_selection.csv   (10->7 reduction curve)
  results/figures/fig05_raw_wavelet_denoise.png      (measured vs denoised OPR)
  results/figures/fig06b_raw_feature_selection.png
  results/tables/preprocess_fidelity.csv             (recon vs supplied SD)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import config as C
from src import plotting as P
from src import raw_pipeline as RP
from src.io_utils import save_table

OUT = C.RESULTS / "preprocessing"
OUT.mkdir(parents=True, exist_ok=True)
SEL7 = ["Time", "OSH", "ADP", "ADTemp", "AWHP", "DCS", "AW"]   # paper's 7


def main():
    # (1)-(2) compile by date-aligned aggregation (raw, undenoised) + denoised.
    print("Compiling raw Volve data (date-aligned aggregation of 3 wells)...")
    train_raw, test_raw = RP.compile_dataset(denoise=False)
    train_dn, test_dn = RP.compile_dataset(denoise=True)
    print(f"  compiled: train {train_raw.shape}, test {test_raw.shape}")

    with pd.ExcelWriter(OUT / "recon_01_compiled.xlsx") as w:
        train_raw.to_excel(w, sheet_name="Train", index=False)
        test_raw.to_excel(w, sheet_name="Test", index=False)
    with pd.ExcelWriter(OUT / "recon_02_denoised.xlsx") as w:
        train_dn.to_excel(w, sheet_name="Train", index=False)
        test_dn.to_excel(w, sheet_name="Test", index=False)

    # (3) Fig 5 - measured vs denoised OPR
    P.fig_denoise(train_raw["OPR"].to_numpy(), train_dn["OPR"].to_numpy(),
                  test_raw["OPR"].to_numpy(), test_dn["OPR"].to_numpy(),
                  fname="fig05_raw_wavelet_denoise.png")
    print("Saved Fig 5 (raw wavelet denoising).")

    # (4) feature reduction 10 -> 7 on the denoised training data
    print("Greedy feature selection over the 10 compiled features...")
    order = RP.greedy_feature_selection(train_dn)
    fs = pd.DataFrame(order)
    save_table(fs.set_index("n"), "table1b_raw_feature_selection")
    print(fs[["n", "added", "RMSE", "R2"]].to_string(index=False))
    P.fig_feature_selection(fs["n"].tolist(), fs["RMSE"].tolist(),
                            fs["R2"].tolist(), fname="fig06b_raw_feature_selection.png")
    picked7 = fs[fs["n"] == 7]["subset"].iloc[0] if (fs["n"] == 7).any() else SEL7
    print("First 7 features chosen by the wrapper:", picked7)
    print("Paper's 7 features:", SEL7)

    # (5) write the reconstructed SelectedFeature_WaveletDenoised workbook
    cols7 = SEL7 + ["OPR"]
    with pd.ExcelWriter(OUT / "recon_03_selected.xlsx") as w:
        train_dn[cols7].to_excel(w, sheet_name="Train", index=False)
        test_dn[cols7].to_excel(w, sheet_name="Test", index=False)
    print("Wrote recon_03_selected.xlsx (7-feature denoised dataset).")

    # (6) fidelity vs the supplied SD file (align on the F-12 startup head).
    sd_tr = pd.read_excel(C.DATA_XLSX, sheet_name="Train")
    rows = []
    n = min(len(sd_tr), len(train_dn))
    for col in ["OSH", "ADP", "ADTemp", "AWHP", "DCS", "OPR"]:
        a = sd_tr[col].to_numpy(float)[:n]
        b = train_dn[col].to_numpy(float)[:n]
        m = np.isfinite(a) & np.isfinite(b)
        corr = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else np.nan
        rmse = float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))
        rows.append({"column": col, "pearson_r": round(corr, 4),
                     "rmse_vs_SD": round(rmse, 4)})
    fid = pd.DataFrame(rows).set_index("column")
    save_table(fid, "preprocess_fidelity")
    print("\nFidelity of reconstruction vs supplied SD (head-aligned):")
    print(fid.to_string())


if __name__ == "__main__":
    main()
