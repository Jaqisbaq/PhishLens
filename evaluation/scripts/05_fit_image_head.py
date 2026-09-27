"""Part A step 4: fit the CLIP image logistic head.

Grouped 5-fold out-of-fold (OOF) predictions on development data (used later
for fusion training), then a final refit on all development data (used for
test-time image scores and shipped as image_head.json).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

import common

CASES = common.CACHE_DIR / "case_table.parquet"
SPLIT = common.CACHE_DIR / "split.parquet"
EMB = common.CACHE_DIR / "image_embeddings.npz"
OUT_OOF = common.CACHE_DIR / "image_oof_dev.parquet"
OUT_HEAD = common.FITTED_DIR / "image_head.json"
OUT_TEST_PROBS = common.CACHE_DIR / "image_test_probs.parquet"

N_FOLDS = 5
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def load_embeddings():
    npz = np.load(EMB, allow_pickle=True)
    ids = npz["ids"].astype(str)
    emb = npz["embeddings"].astype(np.float64)
    return dict(zip(ids, emb))


def main() -> None:
    cases = pd.read_parquet(CASES)[["id", "label", "domain"]]
    split = pd.read_parquet(SPLIT)[["id", "split"]]
    df = cases.merge(split, on="id")
    emb_map = load_embeddings()
    df = df[df["id"].isin(emb_map.keys())].reset_index(drop=True)

    dev = df[df.split == "dev"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)

    X_dev = np.stack([emb_map[i] for i in dev["id"]])
    y_dev = dev["label"].to_numpy()
    groups_dev = dev["domain"].to_numpy()

    X_test = np.stack([emb_map[i] for i in test["id"]])

    # Choose C by grouped CV average precision.
    gkf = GroupKFold(n_splits=N_FOLDS)
    best_c, best_ap = None, -1.0
    for C in C_GRID:
        aps = []
        for tr_idx, va_idx in gkf.split(X_dev, y_dev, groups_dev):
            scaler = StandardScaler().fit(X_dev[tr_idx])
            Xtr = scaler.transform(X_dev[tr_idx])
            Xva = scaler.transform(X_dev[va_idx])
            clf = LogisticRegression(penalty="l2", C=C, max_iter=2000, solver="lbfgs")
            clf.fit(Xtr, y_dev[tr_idx])
            p = clf.predict_proba(Xva)[:, 1]
            aps.append(average_precision_score(y_dev[va_idx], p))
        mean_ap = float(np.mean(aps))
        print(f"C={C:<6} mean AP (grouped 5-fold) = {mean_ap:.4f}")
        if mean_ap > best_ap:
            best_ap, best_c = mean_ap, C
    print("chosen C =", best_c, "AP =", best_ap)

    # OOF predictions on dev with the chosen C, for fusion training.
    oof = np.zeros(len(dev))
    for tr_idx, va_idx in gkf.split(X_dev, y_dev, groups_dev):
        scaler = StandardScaler().fit(X_dev[tr_idx])
        Xtr = scaler.transform(X_dev[tr_idx])
        Xva = scaler.transform(X_dev[va_idx])
        clf = LogisticRegression(penalty="l2", C=best_c, max_iter=2000, solver="lbfgs")
        clf.fit(Xtr, y_dev[tr_idx])
        oof[va_idx] = clf.predict_proba(Xva)[:, 1]

    oof_df = pd.DataFrame({"id": dev["id"], "p_image_head_oof": oof})
    oof_df.to_parquet(OUT_OOF, index=False)

    # Final refit on all development data.
    final_scaler = StandardScaler().fit(X_dev)
    Xdev_s = final_scaler.transform(X_dev)
    final_clf = LogisticRegression(penalty="l2", C=best_c, max_iter=2000, solver="lbfgs")
    final_clf.fit(Xdev_s, y_dev)

    head = {
        "coefficients": [float(c) for c in final_clf.coef_.reshape(-1)],
        "intercept": float(final_clf.intercept_.reshape(-1)[0]),
        "scaler": {
            "mean": [float(v) for v in final_scaler.mean_],
            "scale": [float(v) for v in final_scaler.scale_],
        },
        "fitted": True,
        "fitted_on": "%d development linked cases (%d phishing, %d legitimate), "
                     "grouped 5-fold CV selected C=%g" % (
                         len(dev), int(y_dev.sum()), int(len(y_dev) - y_dev.sum()), best_c),
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cv_mean_average_precision": best_ap,
        "c_grid_searched": C_GRID,
    }
    with open(OUT_HEAD, "w", encoding="utf-8") as f:
        json.dump(head, f, indent=2)
        f.write("\n")

    # Test-time probabilities from the head refit on all of dev.
    Xtest_s = final_scaler.transform(X_test)
    p_test = final_clf.predict_proba(Xtest_s)[:, 1]
    pd.DataFrame({"id": test["id"], "p_image_head": p_test}).to_parquet(OUT_TEST_PROBS, index=False)

    print("wrote", OUT_OOF)
    print("wrote", OUT_HEAD)
    print("wrote", OUT_TEST_PROBS)


if __name__ == "__main__":
    main()
