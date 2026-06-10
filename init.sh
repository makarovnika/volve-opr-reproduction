#!/usr/bin/env bash
# init.sh — environment setup & verification for the Volve OPR reproduction.
# Usage:  bash init.sh        (setup + quick verify)
#         bash init.sh --run  (setup + full pipeline)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "==> Python: $(python --version 2>&1)"

echo "==> Installing dependencies"
python -m pip install -r requirements.txt

echo "==> Verifying inputs"
python - <<'PY'
from pathlib import Path
p = Path("input data/SD28Nov2024_SelectedFeature_WeveletDenoised.xlsx")
assert p.exists(), f"missing input data file: {p}"
import pandas as pd
tr = pd.read_excel(p, sheet_name="Train"); te = pd.read_excel(p, sheet_name="Test")
print(f"   Train {tr.shape}  Test {te.shape}  cols={list(tr.columns)}")
PY

echo "==> Import smoke test"
python - <<'PY'
from src.data import build_dataset
from src.train import fit_model
ds = build_dataset()
res = fit_model("LSTM", ds, stage2_iter=10)
print(f"   LSTM test RMSE={res.test_metrics['RMSE']:.3f}  R2={res.test_metrics['R2']:.3f}")
PY

if [[ "${1:-}" == "--run" ]]; then
  echo "==> Running full pipeline"
  python scripts/run_all.py
fi
echo "==> Done."
