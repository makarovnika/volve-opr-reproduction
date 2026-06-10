"""
Rank-based scoring system (paper Sec. 3.3, Table 6 + Fig. 11 radar plot).

Each metric assigns a rank score from 1 (worst) to N (best, where N = number of
models) to every model, separately for the train and test subsets.  Tied models
share the same (averaged) score, independent of input ordering.  Scores are
summed into a total used to rank the models.

For ARE the comparison is on |ARE| (closeness to zero); for MAE/RMSE/SD lower is
better; for R2 higher is better.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ["ARE", "MAE", "RMSE", "R2", "SD"]
LOWER_BETTER = {"ARE", "MAE", "RMSE", "SD"}     # ARE judged by |ARE|


def _rank_scores(values: np.ndarray, lower_better: bool) -> np.ndarray:
    """Return rank scores in [1, N] where N = best.

    Ties share the SAME (averaged) score, independent of model ordering — so two
    models with identical metrics (e.g. a COA and a PSO hybrid that converge to
    the same solution) get equal scores rather than an arbitrary split decided by
    dict insertion order.  The column sum is preserved (average-rank convention).
    """
    from scipy.stats import rankdata
    values = np.asarray(values, dtype=float)
    n = len(values)
    asc = rankdata(values, method="average")            # 1=smallest .. n=largest
    # lower_better: smallest value is best -> gets score n; largest -> 1.
    # higher_better: largest value is best -> gets score n (== asc).
    return (n + 1 - asc) if lower_better else asc


def score_table(metrics_by_model: dict[str, dict], subset: str) -> pd.DataFrame:
    """``metrics_by_model``: {model_name: {metric: value}} for one subset."""
    models = list(metrics_by_model)
    rows = {}
    for met in METRICS:
        vals = np.array([metrics_by_model[m][met] for m in models])
        if met == "ARE":
            vals = np.abs(vals)
        rows[met] = _rank_scores(vals, lower_better=met in LOWER_BETTER)
    df = pd.DataFrame(rows, index=models)
    df["Score"] = df[METRICS].sum(axis=1)
    df.insert(0, "Subset", subset)
    return df


def combined_scores(train_metrics: dict, test_metrics: dict) -> pd.DataFrame:
    if list(train_metrics) != list(test_metrics):
        raise ValueError("train/test metric dicts must cover the same models "
                         "in the same order to sum their scores.")
    tr = score_table(train_metrics, "Train")
    te = score_table(test_metrics, "Test")
    total = tr["Score"] + te["Score"]
    out = pd.concat([tr, te])
    out = out.sort_index(kind="stable")
    totals = pd.Series(total, name="Total score")
    return out, totals
