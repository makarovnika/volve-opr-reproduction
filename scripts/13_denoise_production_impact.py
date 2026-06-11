"""
Step 13 - Impact of OPR denoising on cumulative oil PRODUCTION (volume).

TASK_denoising_production_impact.md.  OPR is a RATE (BORE_OIL_VOL/ON_STREAM_HRS,
m3 per producing hour), so the denoising effect is expressed in oil VOLUME via

    oil_volume(t) = OPR(t) * ON_STREAM_HRS(t)

  * historical (ground truth) = sum BORE_OIL_VOL(t)         (raw export)
  * denoised production       = sum OPR_denoised(t) * OSH_raw(t)
  * step delta                = oil_raw(t) - oil_denoised(t)   (signed:
        + = denoising shaved production, - = denoising added it)

Two denoising variants (both computed, never silently chosen):
  (1) CODE denoise  - raw_pipeline/config.DENOISE applied per well (OPR column).
  (2) SD FILE       - the released denoised OPR the paper was written on, on the
                      date-aggregated series (sum over wells), OSH = sum OSH.

Honest framing: denoising REDISTRIBUTES the rate series (it does not physically
lose oil); the cumulative delta is a *change in the production estimate*, not
barrels left in the ground.  The current code denoise over-smooths (heavy OPR
setting, see TASK_reconstruct_training_dataset.md) so its delta is an upper bound.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src import raw_pipeline as RP
from src.preprocessing import wavelet_denoise
from src.models import set_seed
from src.io_utils import save_table

M3_TO_BBL = 6.2898                       # 1 m3 = 6.2898 oil barrels
VOLVE_FIELD_BBL = 63e6                   # ~63 Mbbl total Volve recovery (7 wells)


def per_well_raw():
    """Per-well producing-day series: DATE, OSH, BORE_OIL_VOL, OPR_raw."""
    raw = RP.load_raw()
    out = {}
    for well in RP.WELLS:
        s = raw[(raw["NPD_WELL_BORE_NAME"] == well) &
                (raw["ON_STREAM_HRS"] > 0)].sort_values("DATEPRD")
        osh = s["ON_STREAM_HRS"].to_numpy(float)
        vol = s["BORE_OIL_VOL"].to_numpy(float)
        out[well] = pd.DataFrame({
            "date": s["DATEPRD"].to_numpy(),
            "osh": osh,
            "oil_raw": vol,                          # = BORE_OIL_VOL
            "opr_raw": np.where(osh > 0, vol / osh, 0.0),
        })
    return out


def main():
    set_seed(0)
    print(f"OPR convention: rate = BORE_OIL_VOL / ON_STREAM_HRS (m3/hour); "
          f"volume(t) = OPR(t) x OSH(t).")
    print(f"Code denoise = config.DENOISE ({C.DENOISE['wavelet']}/L{C.DENOISE['level']}/"
          f"{C.DENOISE['rule']}/x{C.DENOISE['scale']}); SD denoise = released file.\n")

    wells = per_well_raw()

    # -------- identity checks (acceptance) -------------------------------- #
    max_id_err = 0.0
    for w, d in wells.items():
        recon = d["opr_raw"] * d["osh"]
        max_id_err = max(max_id_err, float(np.max(np.abs(recon - d["oil_raw"]))))
    print(f"[identity] max |OPR_raw*OSH - BORE_OIL_VOL| = {max_id_err:.2e}  "
          f"(expect ~0)")

    # ===================================================================== #
    # VARIANT 1 - CODE denoise, per well
    # ===================================================================== #
    ts_rows, summary = [], []
    for w, d in wells.items():
        opr_cd = wavelet_denoise(d["opr_raw"].to_numpy(float), **C.DENOISE)
        oil_cd = opr_cd * d["osh"].to_numpy(float)
        delta = d["oil_raw"].to_numpy(float) - oil_cd
        sub = pd.DataFrame({
            "date": d["date"], "well": w, "osh": d["osh"],
            "oil_raw": d["oil_raw"], "oil_denoised": oil_cd, "delta_step": delta,
        })
        ts_rows.append(sub)
        n_neg = int((opr_cd < 0).sum())
        neg_vol = float((opr_cd[opr_cd < 0] * d["osh"].to_numpy(float)[opr_cd < 0]).sum())
        imax = int(np.argmax(np.abs(delta)))
        summary.append({
            "well": w, "n_steps": len(d),
            "hist_m3": d["oil_raw"].sum(),
            "denoised_m3": float(oil_cd.sum()),
            "loss_m3": float(delta.sum()),
            "loss_pct": float(100 * delta.sum() / d["oil_raw"].sum()),
            "peak_opr_raw": float(d["opr_raw"].max()),
            "peak_opr_denoised": float(opr_cd.max()),
            "n_negative_opr": n_neg,
            "neg_opr_oil_m3": neg_vol,
            "max_step_delta_m3": float(delta[imax]),
            "max_step_date": str(pd.Timestamp(d["date"][imax]).date()),
        })
    ts = pd.concat(ts_rows, ignore_index=True)

    # TOTAL across wells, by date
    tot = (ts.groupby("date")[["oil_raw", "oil_denoised", "delta_step"]]
           .sum().sort_index())
    tot_hist, tot_den = tot["oil_raw"].sum(), tot["oil_denoised"].sum()
    summary.append({
        "well": "TOTAL", "n_steps": len(ts),
        "hist_m3": float(tot_hist), "denoised_m3": float(tot_den),
        "loss_m3": float(tot_hist - tot_den),
        "loss_pct": float(100 * (tot_hist - tot_den) / tot_hist),
        "peak_opr_raw": float(ts["oil_raw"].max()), "peak_opr_denoised": np.nan,
        "n_negative_opr": int(sum(s["n_negative_opr"] for s in summary)),
        "neg_opr_oil_m3": float(sum(s["neg_opr_oil_m3"] for s in summary)),
        "max_step_delta_m3": float(tot["delta_step"].abs().max()),
        "max_step_date": str(tot["delta_step"].abs().idxmax().date()),
    })
    sm = pd.DataFrame(summary).set_index("well")
    sm["hist_bbl"] = sm["hist_m3"] * M3_TO_BBL
    sm["loss_bbl"] = sm["loss_m3"] * M3_TO_BBL

    # clipping artefact: redo TOTAL with negative OPR clipped to 0
    clip_rows = []
    for w, d in wells.items():
        opr_cd = wavelet_denoise(d["opr_raw"].to_numpy(float), **C.DENOISE)
        oil_clip = np.clip(opr_cd, 0, None) * d["osh"].to_numpy(float)
        clip_rows.append(pd.DataFrame({"date": d["date"], "oil_clip": oil_clip}))
    clip_tot = pd.concat(clip_rows).groupby("date")["oil_clip"].sum().sum()

    # additional: denoise OSH too (OPR & OSH both denoised) -> TOTAL only
    both_tot = 0.0
    for w, d in wells.items():
        opr_cd = wavelet_denoise(d["opr_raw"].to_numpy(float), **C.DENOISE)
        osh_cd = wavelet_denoise(d["osh"].to_numpy(float), **C.DENOISE)
        both_tot += float((opr_cd * osh_cd).sum())

    print("\n=== VARIANT 1: CODE denoise (per well, OSH raw) ===")
    cols = ["n_steps", "hist_m3", "denoised_m3", "loss_m3", "loss_pct",
            "n_negative_opr", "max_step_delta_m3", "max_step_date"]
    print(sm[cols].to_string())
    print(f"\nTOTAL hist  = {tot_hist:,.0f} m3 = {tot_hist*M3_TO_BBL:,.0f} bbl")
    print(f"TOTAL denoised (OSH raw)              = {tot_den:,.0f} m3  "
          f"(loss {tot_hist-tot_den:,.0f} m3, {100*(tot_hist-tot_den)/tot_hist:+.2f}%)")
    print(f"TOTAL denoised (neg OPR clipped to 0) = {clip_tot:,.0f} m3  "
          f"(loss {tot_hist-clip_tot:,.0f} m3, {100*(tot_hist-clip_tot)/tot_hist:+.2f}%)")
    print(f"TOTAL denoised (OPR & OSH denoised)   = {both_tot:,.0f} m3  "
          f"(loss {tot_hist-both_tot:,.0f} m3, {100*(tot_hist-both_tot)/tot_hist:+.2f}%)")

    # sanity check vs field total
    print(f"\n[sanity] 3-well historical = {tot_hist*M3_TO_BBL/1e6:.2f} Mbbl "
          f"= {100*tot_hist*M3_TO_BBL/VOLVE_FIELD_BBL:.1f}% of ~63 Mbbl Volve total.")

    # ===================================================================== #
    # VARIANT 2 - SD FILE denoise, date-aggregated
    # ===================================================================== #
    agg_raw = RP.aggregate_by_date()                       # raw: OPR=sum rates, OSH=sum
    sd = pd.concat([pd.read_excel(C.DATA_XLSX, sheet_name="Train"),
                    pd.read_excel(C.DATA_XLSX, sheet_name="Test")],
                   ignore_index=True)
    n = min(len(agg_raw), len(sd))
    osh_agg = agg_raw["OSH"].to_numpy(float)[:n]
    # IMPORTANT: SD_OPR = sum of per-well rates, so SD_OPR*OSH_agg double-counts
    # the multi-well cross term (sum(rate)*sum(hrs) != sum(rate*hrs)).  To cancel
    # that inflation we put BOTH series on the SAME aggregated-product basis, so
    # the DELTA isolates the denoising effect.  Absolute m3 here are inflated
    # (use Variant 1 for true volumes); only the delta / % are meaningful.
    oil_raw_agg = agg_raw["OPR"].to_numpy(float)[:n] * osh_agg   # raw, same basis
    oil_sd = sd["OPR"].to_numpy(float)[:n] * osh_agg            # SD-denoised, same basis
    delta_sd = oil_raw_agg - oil_sd
    hist2, den2 = float(oil_raw_agg.sum()), float(oil_sd.sum())
    print("\n=== VARIANT 2: SD FILE denoise (date-aggregated rate series) ===")
    print(f"NOTE: aggregated volume = OPR(sum rate) x OSH(sum hrs) is inflated by "
          f"the multi-well cross term; ONLY the delta/%% are meaningful here.")
    print(f"aggregated raw   (inflated basis) = {hist2:,.0f} m3")
    print(f"aggregated SD-denoised            = {den2:,.0f} m3")
    print(f"  delta = {hist2-den2:,.0f} m3 ({100*(hist2-den2)/hist2:+.2f}%) — "
          f"SD denoising effect on the aggregated series")

    # add a Variant-2 TOTAL row to the summary
    sm.loc["TOTAL_SDfile"] = {
        "n_steps": n, "hist_m3": hist2, "denoised_m3": den2,
        "loss_m3": hist2 - den2, "loss_pct": 100*(hist2-den2)/hist2,
        "peak_opr_raw": float(agg_raw["OPR"].max()),
        "peak_opr_denoised": float(sd["OPR"].max()),
        "n_negative_opr": int((sd["OPR"] < 0).sum()),
        "neg_opr_oil_m3": np.nan,
        "max_step_delta_m3": float(delta_sd[np.argmax(np.abs(delta_sd))]),
        "max_step_date": str(pd.Timestamp(agg_raw["DATE"].to_numpy()[np.argmax(np.abs(delta_sd))]).date()),
        "hist_bbl": hist2 * M3_TO_BBL, "loss_bbl": (hist2-den2) * M3_TO_BBL,
    }

    # -------- persist tables --------------------------------------------- #
    save_table(sm.round(2), "denoise_production_impact")
    # full per-step timeseries (variant 1) + cumulative
    ts = ts.sort_values(["well", "date"]).reset_index(drop=True)
    ts["cum_oil_raw"] = ts.groupby("well")["oil_raw"].cumsum()
    ts["cum_oil_denoised"] = ts.groupby("well")["oil_denoised"].cumsum()
    ts["cum_delta"] = ts.groupby("well")["delta_step"].cumsum()
    ts.round(3).to_csv(C.TAB_DIR / "denoise_production_impact_timeseries.csv", index=False)
    print(f"\nWrote denoise_production_impact{{,_timeseries}}.csv")

    # -------- figures ----------------------------------------------------- #
    tot = tot.copy()
    tot["cum_raw"] = tot["oil_raw"].cumsum()
    tot["cum_den"] = tot["oil_denoised"].cumsum()
    tot["cum_delta"] = tot["delta_step"].cumsum()
    dates = tot.index.to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, tot["cum_raw"] / 1e3, label="historical (raw BORE_OIL_VOL)", lw=2)
    ax.plot(dates, tot["cum_den"] / 1e3, label="code-denoised (OPR x OSH)", lw=2, ls="--")
    ax.set_ylabel("cumulative oil ($10^3$ m³)"); ax.set_xlabel("date")
    ax.set_title("Cumulative production: raw vs code-denoised (3 wells)")
    ax.legend(); fig.tight_layout()
    fig.savefig(C.FIG_DIR / "cum_production_raw_vs_denoised.png", dpi=200); plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, tot["cum_delta"] / 1e3, color="tab:red", lw=2)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("cumulative Δ ($10^3$ m³)\n(+ = denoise shaved)"); ax.set_xlabel("date")
    ax.set_title("Cumulative denoising delta over time (code denoise)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "cum_delta_over_time.png", dpi=200); plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, tot["delta_step"], color="tab:purple", lw=0.7)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("step Δ (m³/day)"); ax.set_xlabel("date")
    ax.set_title("Per-step denoising delta (spikes = shaved OPR peaks)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "step_delta_over_time.png", dpi=200); plt.close()
    print("Wrote 3 figures to results/figures/.")


if __name__ == "__main__":
    main()
