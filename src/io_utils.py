"""Small helpers for persisting predictions, metrics and tables between steps."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

PRED_FILE = C.MODEL_DIR / "predictions.npz"
METRIC_FILE = C.MODEL_DIR / "metrics.json"


def save_predictions(measured_train, measured_test, preds_train, preds_test):
    np.savez(
        PRED_FILE,
        measured_train=measured_train,
        measured_test=measured_test,
        **{f"train__{k}": v for k, v in preds_train.items()},
        **{f"test__{k}": v for k, v in preds_test.items()},
    )


def load_predictions():
    d = np.load(PRED_FILE)
    preds_train = {k.split("__", 1)[1]: d[k] for k in d.files if k.startswith("train__")}
    preds_test = {k.split("__", 1)[1]: d[k] for k in d.files if k.startswith("test__")}
    return d["measured_train"], d["measured_test"], preds_train, preds_test


def save_metrics(train_metrics: dict, test_metrics: dict):
    METRIC_FILE.write_text(json.dumps(
        {"train": train_metrics, "test": test_metrics}, indent=2))


def load_metrics():
    d = json.loads(METRIC_FILE.read_text())
    return d["train"], d["test"]


def _to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table without needing tabulate."""
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [fmt(v) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, name: str, float_format="%.4f"):
    p_csv = C.TAB_DIR / f"{name}.csv"
    df.to_csv(p_csv)
    try:
        md = df.to_markdown(floatfmt=".4f")
    except Exception:
        md = _to_markdown(df)
    (C.TAB_DIR / f"{name}.md").write_text(md, encoding="utf-8")
    return p_csv
