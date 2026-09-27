"""Part A steps 5 to 7: fit the learned fusion, compare calibration, and
freeze thresholds, all on development data only.

Uses direct model outputs for the url and text branches (pretrained, not
fit on this data) and the grouped-OOF image head probabilities for the
image branch, with modality-masking augmentation as required by the spec.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, brier_score_loss

import common
common.ensure_src_on_path()
from phishlens import fusion  # noqa: E402

CASES = common.CACHE_DIR / "case_table.parquet"
SPLIT = common.CACHE_DIR / "split.parquet"
URL_P = common.CACHE_DIR / "url_predictions.parquet"
TEXT_P = common.CACHE_DIR / "text_predictions.parquet"
IMG_OOF = common.CACHE_DIR / "image_oof_dev.parquet"
IMG_TEST = common.CACHE_DIR / "image_test_probs.parquet"

OUT_FUSION = common.FITTED_DIR / "fusion.json"
OUT_DEV_OOF = common.CACHE_DIR / "fusion_dev_oof.parquet"

N_FOLDS = 5
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
BRANCHES = ("url", "text", "image")


def load_frame():
    cases = pd.read_parquet(CASES)[["id", "label", "domain"]]
    split = pd.read_parquet(SPLIT)[["id", "split"]]
    url = pd.read_parquet(URL_P)[["id", "p_url"]]
    text = pd.read_parquet(TEXT_P)[["id", "p_text"]]
    df = cases.merge(split, on="id").merge(url, on="id").merge(text, on="id")
    return df


def augment(P, y, groups):
    """Return augmented (X, y, groups) with each row also present with one
    and with two branches masked (availability 0, feature contribution 0)."""
    n = P.shape[0]
    avail_full = np.ones((n, 3), dtype=bool)
    variants = [avail_full]
    for i in range(3):
        a = avail_full.copy()
        a[:, i] = False
        variants.append(a)  # one branch masked
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        a = avail_full.copy()
        a[:, i] = False
        a[:, j] = False
        variants.append(a)  # two branches masked
    Xs, ys, gs = [], [], []
    for a in variants:
        Xs.append(fusion.build_feature_matrix(P, a))
        ys.append(y)
        gs.append(groups)
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def cv_select_c(P, y, groups):
    gkf = GroupKFold(n_splits=N_FOLDS)
    best_c, best_ap = None, -1.0
    results = {}
    for C in C_GRID:
        aps = []
        for tr_idx, va_idx in gkf.split(P, y, groups):
            Xtr, ytr, gtr = augment(P[tr_idx], y[tr_idx], groups[tr_idx])
            clf = LogisticRegression(penalty="l2", C=C, max_iter=2000, solver="lbfgs")
            clf.fit(Xtr, ytr)
            Xva = fusion.build_feature_matrix(P[va_idx], np.ones((len(va_idx), 3), dtype=bool))
            p = clf.predict_proba(Xva)[:, 1]
            aps.append(average_precision_score(y[va_idx], p))
        mean_ap = float(np.mean(aps))
        results[C] = mean_ap
        print(f"[fusion cv] C={C:<6} mean AP = {mean_ap:.4f}")
        if mean_ap > best_ap:
            best_ap, best_c = mean_ap, C
    return best_c, best_ap, results


def oof_learned_scores(P, y, groups, C):
    """OOF fused probability for every dev row, all three branches available
    (matches deployment: full-case scoring), used for calibration/thresholds."""
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.full(len(y), np.nan)
    for tr_idx, va_idx in gkf.split(P, y, groups):
        Xtr, ytr, _ = augment(P[tr_idx], y[tr_idx], groups[tr_idx])
        clf = LogisticRegression(penalty="l2", C=C, max_iter=2000, solver="lbfgs")
        clf.fit(Xtr, ytr)
        Xva = fusion.build_feature_matrix(P[va_idx], np.ones((len(va_idx), 3), dtype=bool))
        oof[va_idx] = clf.predict_proba(Xva)[:, 1]
    return oof


def fit_platt(z, y):
    lr = LogisticRegression(max_iter=2000)
    lr.fit(z.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def fit_temperature(z, y):
    # simple 1-D search minimising NLL
    best_t, best_nll = 1.0, np.inf
    for t in np.linspace(0.2, 5.0, 97):
        p = 1.0 / (1.0 + np.exp(-z / t))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best_nll:
            best_nll, best_t = nll, t
    return best_t


def choose_thresholds(scores, y):
    order = np.unique(scores)
    # high: lowest threshold with FPR <= 5%
    high = None
    for t in np.sort(order):
        pred = (scores >= t).astype(int)
        fp = np.sum((pred == 1) & (y == 0))
        neg = np.sum(y == 0)
        fpr = fp / neg if neg else 0.0
        if fpr <= 0.05:
            high = float(t)
            break
    if high is None:
        high = float(order.max())
    # low: highest threshold where recall of phishing ABOVE it is >= 95%
    low = None
    for t in np.sort(order)[::-1]:
        pred = (scores >= t).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        pos = np.sum(y == 1)
        recall = tp / pos if pos else 0.0
        if recall >= 0.95:
            low = float(t)
            break
    if low is None:
        low = float(order.min())
    if low >= high:
        low = max(0.0, high - 0.01)
    return round(low, 4), round(high, 4)


def main() -> None:
    df = load_frame()
    img_oof = pd.read_parquet(IMG_OOF)
    df = df.merge(img_oof, on="id", how="left")

    dev = df[df.split == "dev"].reset_index(drop=True)
    P = dev[["p_url", "p_text", "p_image_head_oof"]].to_numpy(dtype=np.float64)
    # image_head_oof should be present for every dev row (all dev rows have screenshots)
    assert not np.isnan(P[:, 2]).any(), "missing OOF image probabilities for some dev rows"
    y = dev["label"].to_numpy()
    groups = dev["domain"].to_numpy()

    best_c, best_ap, cv_table = cv_select_c(P, y, groups)
    print("chosen C =", best_c, "mean AP =", best_ap)

    oof_learned = oof_learned_scores(P, y, groups, best_c)

    # --- calibration comparison (on OOF dev scores) ---
    z = np.array([fusion.clipped_logit(p) for p in oof_learned])
    brier_uncal = brier_score_loss(y, oof_learned)

    plat_a, plat_b = fit_platt(z, y)
    p_platt = 1.0 / (1.0 + np.exp(-(plat_a * z + plat_b)))
    brier_platt = brier_score_loss(y, p_platt)

    temp_t = fit_temperature(z, y)
    p_temp = 1.0 / (1.0 + np.exp(-z / temp_t))
    brier_temp = brier_score_loss(y, p_temp)

    print(f"Brier uncalibrated={brier_uncal:.4f} platt={brier_platt:.4f} temp={brier_temp:.4f}")

    calibration = None
    calibrated_scores = oof_learned
    chosen_cal_name = "none"
    best_brier = brier_uncal
    if brier_platt < best_brier:
        best_brier = brier_platt
        calibration = {"type": "platt", "a": plat_a, "b": plat_b}
        calibrated_scores = p_platt
        chosen_cal_name = "platt"
    if brier_temp < best_brier:
        best_brier = brier_temp
        calibration = {"type": "temperature", "temperature": temp_t}
        calibrated_scores = p_temp
        chosen_cal_name = "temperature"

    print("calibration chosen:", chosen_cal_name, "dev OOF Brier:", best_brier)

    low, high = choose_thresholds(calibrated_scores, y)

    # disagreement_gap rule: on development data, compute the gap between the
    # highest and lowest available branch probability for every case, then set
    # disagreement_gap at the 90th percentile of that distribution, so the
    # disagreement gate fires on roughly the most-disagreeing 10 percent of
    # development cases (a fixed, data-driven rule, not a guess).
    gaps = P.max(axis=1) - P.min(axis=1)
    disagreement_gap = round(float(np.percentile(gaps, 90)), 4)
    print("dev branch-gap distribution: min=%.3f median=%.3f p90=%.3f max=%.3f" % (
        gaps.min(), np.median(gaps), disagreement_gap, gaps.max()))
    print("thresholds low/high:", low, high, "disagreement_gap:", disagreement_gap)

    # --- final refit on all of dev (augmented) for the shipped model ---
    Xall, yall, _ = augment(P, y, groups)
    final_clf = LogisticRegression(penalty="l2", C=best_c, max_iter=2000, solver="lbfgs")
    final_clf.fit(Xall, yall)

    cfg = fusion.fit(
        probabilities=P,
        availability=np.ones_like(P, dtype=bool),
        labels=y,
        C=best_c,
        thresholds={"low": low, "high": high},
        disagreement_gap=disagreement_gap,
        calibration=calibration,
        fitted_on="%d development linked cases (%d phishing, %d legitimate), url+text direct "
                  "model output, image branch grouped 5-fold out-of-fold logistic head, "
                  "modality-masking augmentation, C=%g chosen by grouped 5-fold CV mean "
                  "average precision (%.4f)" % (len(y), int(y.sum()), int(len(y) - y.sum()),
                                                  best_c, best_ap),
        notes="Calibration comparison on development OOF predictions: Brier "
              "uncalibrated=%.4f, platt=%.4f, temperature=%.4f. Chosen: %s. "
              "disagreement_gap set at the 90th percentile of the development "
              "max-minus-min branch probability gap (%.4f), so the disagreement "
              "gate fires on roughly the most-disagreeing 10 percent of "
              "development cases." % (
                  brier_uncal, brier_platt, brier_temp, chosen_cal_name, disagreement_gap),
        output_path=OUT_FUSION,
        write=True,
    )
    # fit() above already writes using its own logistic regression fit inside;
    # but that duplicate fit doesn't include the same augmentation call trace
    # explicitly, so overwrite the coefficients with our own augmented fit to
    # guarantee consistency with the CV procedure above.
    cfg.coefficients = [float(c) for c in final_clf.coef_.reshape(-1)]
    cfg.intercept = float(final_clf.intercept_.reshape(-1)[0])
    cfg.validate()
    fusion.save_config(cfg, OUT_FUSION)

    pd.DataFrame({
        "id": dev["id"], "label": y, "p_url": P[:, 0], "p_text": P[:, 1],
        "p_image_head_oof": P[:, 2], "learned_oof_raw": oof_learned,
        "learned_oof_calibrated": calibrated_scores,
    }).to_parquet(OUT_DEV_OOF, index=False)

    print("wrote", OUT_FUSION)
    print("wrote", OUT_DEV_OOF)


if __name__ == "__main__":
    main()
