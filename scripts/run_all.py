"""
Run the full reproduction pipeline end to end.

Usage:
    python scripts/run_all.py            # fast path (Table-2 architectures)
    python scripts/run_all.py --full     # also run Stage-1 PSO search (Fig. 7)

Steps:
    [00] Reconstruct preprocessing from RAW Volve data (Fig. 5)  [--raw only]
    01 EDA + correlation (Fig. 4, Fig. 2)
    02 NSGA-II feature selection (Table 1, Fig. 6)
    [00] Stage-1 PSO structure search (Fig. 7)          [--full only]
    03 Train 6 models (Tables 3-4, Figs 8-10)
    04 Scoring + radar + time series (Table 6, Figs 11-12)
    05 Bootstrap (Table 5)
    06 SHAP (Fig. 13)
    07 Sensitivity scenarios (Fig. 14)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FULL = "--full" in sys.argv
RAW = "--raw" in sys.argv      # also reconstruct preprocessing from raw Volve data


def _run(label, fn):
    print("\n" + "=" * 70)
    print(f">>> {label}")
    print("=" * 70)
    t0 = time.time()
    fn()
    print(f"--- {label} done in {time.time() - t0:.1f}s")


def main():
    import importlib

    def step(modname, **kw):
        mod = importlib.import_module(f"scripts.{modname}")
        return lambda: mod.main(**kw)

    if RAW:
        _run("00 Reconstruct preprocessing from raw", step("00_preprocess"))
        _run("00b Reconstruct denoising regimes", step("00b_denoising"))
        _run("00c Validate reconstruction vs SD", step("00c_validate_reconstruction"))
    _run("01 EDA", step("01_explore_data"))
    _run("02 Feature selection", step("02_feature_selection"))
    if FULL:
        _run("00 Stage-1 PSO structure (Fig. 7)", step("00_stage1_structure"))
    _run("03 Train models", step("03_train_models"))
    _run("04 Evaluate", step("04_evaluate"))
    _run("05 Bootstrap", step("05_bootstrap"))
    _run("06 SHAP", step("06_shap"))
    _run("07 Sensitivity", step("07_sensitivity"))
    _run("08 Paper comparison", step("08_paper_comparison"))
    _run("10 Dual-target (denoised + raw OPR)", step("10_dual_target"))
    _run("12 Optimizer-lift ablation", step("12_ablation"))
    if FULL:
        _run("11 Seed ensemble + variance", step("11_seed_ensemble"))
    if RAW:
        _run("09 Reconstruction validation", step("09_recon_validation"))
    print("\nAll steps complete. See results/figures and results/tables.")


if __name__ == "__main__":
    main()
