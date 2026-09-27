"""Part A steps 8 to 11: the one-time locked test evaluation.

Computes, for every method (url, text, image, mean, max, learned):
precision, recall, F1, FPR, FN count, ROC AUC, average precision, Brier,
expected calibration error (10 bins), confusion counts at 0.5 and at the
frozen thresholds, and group-level bootstrap 95 percent intervals. Also the
three-outcome table, the ablation, and the error inspection list.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve,
)

import common
common.ensure_src_on_path()
from phishlens import fusion  # noqa: E402
from phishlens import policy  # noqa: E402
from phishlens.fusion import FusionConfig  # noqa: E402

CASES = common.CACHE_DIR / "case_table.parquet"
SPLIT = common.CACHE_DIR / "split.parquet"
URL_P = common.CACHE_DIR / "url_predictions.parquet"
TEXT_P = common.CACHE_DIR / "text_predictions.parquet"
IMG_TEST = common.CACHE_DIR / "image_test_probs.parquet"
IMG_ZS = common.CACHE_DIR / "image_zeroshot.parquet"
FUSION_JSON = common.FITTED_DIR / "fusion.json"
DEV_OOF = common.CACHE_DIR / "fusion_dev_oof.parquet"

OUT_RESULTS = common.OUTPUTS_DIR / "results.json"
OUT_TEST_METRICS = common.TABLES_DIR / "test_metrics.csv"
OUT_THREE_OUTCOME = common.TABLES_DIR / "three_outcome.csv"
OUT_ABLATION = common.TABLES_DIR / "ablation.csv"
OUT_PRED_CACHE = common.CACHE_DIR / "test_all_predictions.parquet"

N_BOOT = 1000
SEED = common.SEED
METHODS = ["url", "text", "image", "mean", "max", "learned"]


def ece(y, p, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins[1:-1], right=True)
    total = len(y)
    err = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        err += (mask.sum() / total) * abs(acc - conf)
    return float(err)


def confusion(y, pred):
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def point_metrics(y, scores):
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    pred50 = (scores >= 0.5).astype(int)
    out = {
        "n": int(len(y)),
        "precision_at_0.5": float(precision_score(y, pred50, zero_division=0)),
        "recall_at_0.5": float(recall_score(y, pred50, zero_division=0)),
        "f1_at_0.5": float(f1_score(y, pred50, zero_division=0)),
        "fpr_at_0.5": confusion(y, pred50)["fp"] / max(1, int((y == 0).sum())),
        "fn_count_at_0.5": confusion(y, pred50)["fn"],
        "confusion_at_0.5": confusion(y, pred50),
    }
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, scores))
    else:
        out["roc_auc"] = None
    out["average_precision"] = float(average_precision_score(y, scores))
    out["brier"] = float(brier_score_loss(y, scores))
    out["ece_10bin"] = ece(y, scores)
    return out


def _group_index_lists(domains):
    """Map each unique domain to the row indices that belong to it (numpy, fast)."""
    uniq, inverse = np.unique(domains, return_inverse=True)
    idx_lists = [[] for _ in uniq]
    for row_i, g_i in enumerate(inverse):
        idx_lists[g_i].append(row_i)
    return uniq, [np.array(x, dtype=np.int64) for x in idx_lists]


def bootstrap_group_ci(df, score_col, metric_fn, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    y_all = df["label"].to_numpy()
    s_all = df[score_col].to_numpy()
    uniq, idx_lists = _group_index_lists(df["domain"].to_numpy())
    n_groups = len(uniq)
    vals = []
    for _ in range(n_boot):
        chosen = rng.randint(0, n_groups, n_groups)
        idx = np.concatenate([idx_lists[c] for c in chosen])
        try:
            vals.append(metric_fn(y_all[idx], s_all[idx]))
        except Exception:
            continue
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_group_bootstrap_diff(df, col_a, col_b, metric_fn, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    y_all = df["label"].to_numpy()
    a_all = df[col_a].to_numpy()
    b_all = df[col_b].to_numpy()
    uniq, idx_lists = _group_index_lists(df["domain"].to_numpy())
    n_groups = len(uniq)
    diffs = []
    for _ in range(n_boot):
        chosen = rng.randint(0, n_groups, n_groups)
        idx = np.concatenate([idx_lists[c] for c in chosen])
        try:
            ma = metric_fn(y_all[idx], a_all[idx])
            mb = metric_fn(y_all[idx], b_all[idx])
        except Exception:
            continue
        diffs.append(ma - mb)
    diffs = np.array(diffs)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_le_0": float(np.mean(diffs <= 0)),
    }


def outcome_for(score, branch_probs, cfg):
    if score is not None and not np.isfinite(score):
        score = None
    decision = policy.decide(score, branch_probs, cfg.thresholds["low"], cfg.thresholds["high"],
                              cfg.disagreement_gap)
    return decision.outcome


def main() -> None:
    cases = pd.read_parquet(CASES)[["id", "label", "domain", "url"]]
    split = pd.read_parquet(SPLIT)[["id", "split"]]
    url = pd.read_parquet(URL_P)[["id", "p_url"]]
    text = pd.read_parquet(TEXT_P)[["id", "p_text"]]
    img = pd.read_parquet(IMG_TEST)[["id", "p_image_head"]]
    zs = pd.read_parquet(IMG_ZS)[["id", "p_image_zero_shot"]]

    df = cases.merge(split, on="id").merge(url, on="id").merge(text, on="id")
    df = df.merge(img, on="id", how="left").merge(zs, on="id", how="left")
    test = df[df.split == "test"].reset_index(drop=True)
    assert test["p_image_head"].notna().all(), "missing fitted-head image probability for some test cases"

    cfg = fusion.load_config(FUSION_JSON)
    assert cfg.fitted, "fusion.json is not marked fitted; refusing to run the one-time test evaluation"

    P = test[["p_url", "p_text", "p_image_head"]].to_numpy(dtype=np.float64)
    avail = np.ones_like(P, dtype=bool)

    scores = {}
    for method in METHODS:
        if method in ("url", "text", "image"):
            col = {"url": "p_url", "text": "p_text", "image": "p_image_head"}[method]
            scores[method] = test[col].to_numpy(dtype=np.float64)
        else:
            scores[method] = fusion.fuse_batch(P, avail, cfg=cfg, method=method)

    test = test.assign(**{f"score_{m}": scores[m] for m in METHODS})
    test.to_parquet(OUT_PRED_CACHE, index=False)

    # pick best single branch on DEVELOPMENT data (not test) by average precision
    dev_oof = pd.read_parquet(DEV_OOF)
    dev_ap = {
        "url": average_precision_score(dev_oof["label"], dev_oof["p_url"]),
        "text": average_precision_score(dev_oof["label"], dev_oof["p_text"]),
        "image": average_precision_score(dev_oof["label"], dev_oof["p_image_head_oof"]),
    }
    best_branch = max(dev_ap, key=dev_ap.get)
    print("dev AP per branch:", dev_ap, "-> best single branch:", best_branch)

    y = test["label"].to_numpy()
    results = {"methods": {}, "best_single_branch_selected_on_dev": best_branch,
               "dev_average_precision_per_branch": dev_ap}
    rows_for_csv = []
    for m in METHODS:
        pm = point_metrics(y, scores[m])
        ci_ap = bootstrap_group_ci(test.assign(_s=scores[m]).rename(columns={"_s": "sc"}),
                                    "sc", lambda yy, ss: average_precision_score(yy, ss))
        ci_f1 = bootstrap_group_ci(test.assign(_s=scores[m]).rename(columns={"_s": "sc"}),
                                    "sc", lambda yy, ss: f1_score(yy, (ss >= 0.5).astype(int), zero_division=0))
        pm["average_precision_ci95"] = ci_ap
        pm["f1_at_0.5_ci95"] = ci_f1
        # confusion at frozen thresholds (high threshold as the "flag positive" cut)
        pred_frozen = (scores[m] >= cfg.thresholds["high"]).astype(int)
        pm["confusion_at_frozen_high_threshold"] = confusion(y, pred_frozen)
        pm["frozen_thresholds"] = cfg.thresholds
        results["methods"][m] = pm
        rows_for_csv.append({"method": m, "n": pm["n"], "precision_0.5": pm["precision_at_0.5"],
                              "recall_0.5": pm["recall_at_0.5"], "f1_0.5": pm["f1_at_0.5"],
                              "fpr_0.5": pm["fpr_at_0.5"], "fn_count_0.5": pm["fn_count_at_0.5"],
                              "roc_auc": pm["roc_auc"], "average_precision": pm["average_precision"],
                              "ap_ci_low": ci_ap[0], "ap_ci_high": ci_ap[1],
                              "brier": pm["brier"], "ece_10bin": pm["ece_10bin"]})
    pd.DataFrame(rows_for_csv).to_csv(OUT_TEST_METRICS, index=False)

    # Paired bootstrap difference: learned vs best single branch
    test2 = test.copy()
    test2["_learned"] = scores["learned"]
    test2["_best_single"] = scores[best_branch]
    paired_ap = paired_group_bootstrap_diff(test2, "_learned", "_best_single",
                                             lambda yy, ss: average_precision_score(yy, ss))
    paired_f1 = paired_group_bootstrap_diff(
        test2, "_learned", "_best_single",
        lambda yy, ss: f1_score(yy, (ss >= 0.5).astype(int), zero_division=0))
    results["paired_bootstrap_learned_vs_best_single"] = {
        "best_single_branch": best_branch,
        "average_precision_diff": paired_ap,
        "f1_diff": paired_f1,
    }

    # --- three-outcome table (step 9), on the learned fusion ---
    outcomes = [
        outcome_for(s, {"url": pu, "text": pt, "image": pim}, cfg)
        for s, pu, pt, pim in zip(scores["learned"], test["p_url"], test["p_text"], test["p_image_head"])
    ]
    test3 = test.assign(outcome=outcomes)
    three = test3.groupby(["outcome", "label"]).size().unstack(fill_value=0)
    for lbl in (0, 1):
        if lbl not in three.columns:
            three[lbl] = 0
    three = three.rename(columns={0: "legitimate", 1: "phishing"})
    three = three.reindex(["High concern", "Review needed", "Limited indicators", "Analysis unavailable"],
                           fill_value=0)
    three["total"] = three["legitimate"] + three["phishing"]
    three["coverage"] = three["total"] / len(test3)
    three.to_csv(OUT_THREE_OUTCOME)
    review_rate = float((test3.outcome == "Review needed").mean())
    phish_limited = int(((test3.outcome == "Limited indicators") & (test3.label == 1)).sum())
    results["three_outcome"] = {
        "table": three.reset_index().to_dict(orient="records"),
        "review_rate": review_rate,
        "phishing_given_limited_indicators": phish_limited,
    }

    # --- ablation (step 10): learned fusion with branches masked ---
    ablation_rows = []
    masks = {
        "full (no masking)": [True, True, True],
        "mask url": [False, True, True],
        "mask text": [True, False, True],
        "mask image": [True, True, False],
        "mask url+text": [False, False, True],
        "mask url+image": [False, True, False],
        "mask text+image": [True, False, False],
    }
    for name, m in masks.items():
        avail_m = np.tile(np.array(m, dtype=bool), (len(test), 1))
        s = fusion.fuse_batch(P, avail_m, cfg=cfg, method="learned")
        finite = np.isfinite(s)
        pm = point_metrics(y[finite], s[finite])
        ci_ap = bootstrap_group_ci(test[finite].assign(_s=s[finite]).rename(columns={"_s": "sc"}),
                                    "sc", lambda yy, ss: average_precision_score(yy, ss))
        ablation_rows.append({"branches_masked": name, "n_scored": int(finite.sum()),
                               "average_precision": pm["average_precision"],
                               "ap_ci_low": ci_ap[0], "ap_ci_high": ci_ap[1],
                               "f1_at_0.5": pm["f1_at_0.5"], "roc_auc": pm["roc_auc"],
                               "brier": pm["brier"]})
    pd.DataFrame(ablation_rows).to_csv(OUT_ABLATION, index=False)
    results["ablation"] = ablation_rows

    # --- error inspection (step 11) ---
    test4 = test.copy()
    test4["learned_score"] = scores["learned"]
    test4["error_magnitude"] = np.where(test4.label == 1, 1 - test4.learned_score, test4.learned_score)
    worst = test4.sort_values("error_magnitude", ascending=False).head(6)

    def defang(u: str) -> str:
        return u.replace("http://", "hxxp://").replace("https://", "hxxps://").replace(".", "[.]")

    errors = []
    for _, r in worst.iterrows():
        errors.append({
            "id": r["id"], "label": "phishing" if r["label"] == 1 else "legitimate",
            "url_defanged": defang(r["url"]) if "url" in r else None,
            "p_url": float(r["p_url"]), "p_text": float(r["p_text"]),
            "p_image_head": float(r["p_image_head"]),
            "fused_learned_score": float(r["learned_score"]),
        })
    results["error_inspection_worst_6"] = errors

    results["fusion_config_used"] = cfg.to_dict()
    results["n_test"] = int(len(test))
    results["n_dev"] = None  # filled by run_manifest

    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
        f.write("\n")

    print("wrote", OUT_RESULTS)
    print("wrote", OUT_TEST_METRICS)
    print("wrote", OUT_THREE_OUTCOME)
    print("wrote", OUT_ABLATION)
    print("wrote", OUT_PRED_CACHE)


if __name__ == "__main__":
    main()
