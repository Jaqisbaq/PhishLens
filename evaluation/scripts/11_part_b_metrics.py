"""Part B metrics: URL branch on the PhiUSIIL sample and text branch on the
CEAS_08 sample, same metric set as Part A, plain (row-level, not grouped)
bootstrap intervals since these are independent single-item samples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, precision_score,
    recall_score, roc_auc_score,
)

import common

URL_PREDS = common.CACHE_DIR / "partb_url_predictions.parquet"
TEXT_PREDS = common.CACHE_DIR / "partb_text_predictions.parquet"
OUT_TABLE = common.TABLES_DIR / "partb_component_metrics.csv"

N_BOOT = 1000
SEED = common.SEED


def ece(y, p, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins[1:-1], right=True)
    total = len(y)
    err = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        err += (mask.sum() / total) * abs(y[mask].mean() - p[mask].mean())
    return float(err)


def plain_bootstrap_ci(y, p, metric_fn, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            vals.append(metric_fn(y[idx], p[idx]))
        except Exception:
            continue
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def compute(name, y, p):
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= 0.5).astype(int)
    ap = average_precision_score(y, p)
    ci_ap = plain_bootstrap_ci(y, p, lambda yy, pp: average_precision_score(yy, pp))
    ci_f1 = plain_bootstrap_ci(y, p, lambda yy, pp: f1_score(yy, (pp >= 0.5).astype(int), zero_division=0))
    return {
        "component": name, "n": len(y),
        "precision_0.5": float(precision_score(y, pred, zero_division=0)),
        "recall_0.5": float(recall_score(y, pred, zero_division=0)),
        "f1_0.5": float(f1_score(y, pred, zero_division=0)),
        "fpr_0.5": float(np.sum((pred == 1) & (y == 0)) / max(1, np.sum(y == 0))),
        "fn_count_0.5": int(np.sum((pred == 0) & (y == 1))),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": ap, "ap_ci_low": ci_ap[0], "ap_ci_high": ci_ap[1],
        "f1_ci_low": ci_f1[0], "f1_ci_high": ci_f1[1],
        "brier": float(brier_score_loss(y, p)),
        "ece_10bin": ece(y, p),
    }


def main() -> None:
    rows = []
    url = pd.read_parquet(URL_PREDS)
    rows.append(compute("url (PhiUSIIL, n=20000 stratified sample, leakage removed)",
                         url["label"], url["p_url"]))
    text = pd.read_parquet(TEXT_PREDS)
    rows.append(compute("text (CEAS_08, 500 phishing / 500 legitimate)",
                         text["label"], text["p_text"]))
    pd.DataFrame(rows).to_csv(OUT_TABLE, index=False)
    print(pd.DataFrame(rows))
    print("wrote", OUT_TABLE)


if __name__ == "__main__":
    main()
