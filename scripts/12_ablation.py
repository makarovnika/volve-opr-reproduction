"""
Step 12 - Honest optimizer-lift ablation (Task 5).

Separates the two contributions that the "hybrid > standalone" gap conflates:
  * BUDGET  — training the standalone DL longer (130 → 600 epochs);
  * STAGE-2 — the metaheuristic (COA/PSO) weight refinement on top.

Two regimes per architecture × optimizer:
  A. equal-budget : standalone(600 ep) vs +Stage-2  → gap = PURE Stage-2 effect
  B. paper-style  : weak standalone(130 ep) + LARGER Stage-2 (2× iters, 2× bounds)

Output: results/tables/table7_ablation.csv / .md
"""
import sys
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import config as C
from src import metrics as M
from src import io_utils as IO
from src.data import build_dataset
from src.models import build_model, set_seed
from src.train import train_backprop, _stage2_refine, predict_norm, _S2_TAIL_FAMILIES


def main():
    ds = build_dataset()
    n_in, seq_len = ds.X_train.shape[-1], ds.X_train.shape[1]

    def rmse(model):
        return M.rmse(ds.y_test_phys, ds.inv_test(predict_norm(model, ds.X_test)))

    def base(arch, epochs):
        # Use the SAME early-stopping val split the main pipeline uses for this
        # architecture (tail for the transfer-fragile CNN, random for the LSTM).
        val_mode = "tail" if arch in _S2_TAIL_FAMILIES else "random"
        set_seed(C.SEED)
        m = build_model(arch, n_in, seq_len=seq_len)
        train_backprop(m, ds.X_train, ds.y_train, epochs=epochs, val_mode=val_mode)
        return m

    rows = []
    for arch in ["CNN", "LSTM"]:
        print(f"== {arch} ==")
        m130 = base(arch, 130); r130 = rmse(m130)
        m600 = base(arch, 600); r600 = rmse(m600)
        print(f"  standalone 130ep={r130:.3f}  600ep={r600:.3f}")
        for method in ["COA", "PSO"]:
            # A: equal budget (600) + normal Stage-2  -> pure Stage-2 lift
            mA = copy.deepcopy(m600)
            _stage2_refine(mA, ds, method, 60, 0.15, np.random.default_rng(C.SEED))
            rA = rmse(mA)
            # B: weak standalone (130) + LARGER Stage-2 (2x iters, 2x bounds)
            mB = copy.deepcopy(m130)
            _stage2_refine(mB, ds, method, 120, 0.30, np.random.default_rng(C.SEED))
            rB = rmse(mB)
            rows.append({
                "arch": arch, "optimizer": method,
                "standalone_130": round(r130, 3),
                "standalone_600": round(r600, 3),
                "A: 600+Stage2": round(rA, 3),
                "pure_Stage2_lift": round(r600 - rA, 3),     # +ve = Stage-2 helped
                "B: 130+largeStage2": round(rB, 3),
                "budget_lift(130->600)": round(r130 - r600, 3),
            })
            print(f"  {method}: A(600+S2)={rA:.3f} (pureS2 lift {r600-rA:+.3f})  "
                  f"B(130+largeS2)={rB:.3f}")

    tab = pd.DataFrame(rows).set_index(["arch", "optimizer"])
    IO.save_table(tab.reset_index().set_index("arch"), "table7_ablation")
    print("\n" + tab.to_string())
    pure = tab["pure_Stage2_lift"].mean()
    budget = tab["budget_lift(130->600)"].mean()
    print(f"\nMean PURE Stage-2 lift: {pure:+.3f} m³/day   "
          f"Mean BUDGET lift (130->600): {budget:+.3f} m³/day")
    print("Interpretation: the hybrid>standalone gap is dominated by the EPOCH "
          "budget; the metaheuristic Stage-2 alone adds ~0 (and can hurt the "
          "transfer-fragile CNN). Matches docs/CRITICAL_REVIEW.md.")


if __name__ == "__main__":
    main()
