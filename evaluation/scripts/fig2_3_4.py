"""Figures 2 to 4: precision-recall curves, confusion of the learned fusion
at frozen thresholds (three-outcome by label), and the reliability diagram
before/after calibration.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

import common
import plotting as pl
import matplotlib.pyplot as plt

TEST_PRED = common.CACHE_DIR / "test_all_predictions.parquet"
FUSION_JSON = common.FITTED_DIR / "fusion.json"
DEV_OOF = common.CACHE_DIR / "fusion_dev_oof.parquet"
THREE_OUTCOME = common.TABLES_DIR / "three_outcome.csv"


def fig2_pr_curves():
    df = pd.read_parquet(TEST_PRED)
    y = df["label"].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for method in ["url", "text", "image", "mean", "max", "learned"]:
        p, r, _ = precision_recall_curve(y, df[f"score_{method}"].to_numpy())
        ax.plot(r, p, label=method, color=pl.BRANCH_COLORS.get(method), linewidth=2)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower left")
    pl.save(fig, common.FIGURES_DIR / "fig2_precision_recall_test.png")


def fig3_confusion_three_outcome():
    three = pd.read_csv(THREE_OUTCOME)
    outcomes = list(three["outcome"])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(outcomes))
    w = 0.38
    ax.bar(x - w / 2, three["phishing"], w, label="phishing", color=pl.PALETTE["vermillion"])
    ax.bar(x + w / 2, three["legitimate"], w, label="legitimate", color=pl.PALETTE["blue"])
    ax.set_xticks(x)
    ax.set_xticklabels(outcomes, rotation=15, ha="right")
    ax.set_ylabel("test case count")
    ax.legend(frameon=False)
    pl.save(fig, common.FIGURES_DIR / "fig3_three_outcome_by_label_test.png")


def fig4_reliability():
    df = pd.read_parquet(TEST_PRED)
    y = df["label"].to_numpy()
    cfg = json.load(open(FUSION_JSON))
    raw_col = "score_learned"

    # "before calibration" = raw (uncalibrated) fused probability recomputed
    # without the calibration step; approximate with sigmoid(logit) via the
    # dev OOF raw column's calibration mapping inverted is not exact, so
    # instead recompute directly: if a calibration is in force, undo it.
    calibrated = df[raw_col].to_numpy()
    calibration = cfg.get("calibration")
    if calibration:
        eps = 1e-6
        z_cal = np.log(np.clip(calibrated, eps, 1 - eps) / (1 - np.clip(calibrated, eps, 1 - eps)))
        if calibration["type"] == "temperature":
            z_raw = z_cal * calibration["temperature"]
        else:
            z_raw = (z_cal - calibration["b"]) / calibration["a"]
        raw = 1.0 / (1.0 + np.exp(-z_raw))
    else:
        raw = calibrated

    fig, ax = plt.subplots(figsize=(6, 6))
    for scores, label, color in [(raw, "before calibration", pl.PALETTE["grey"]),
                                  (calibrated, "after calibration" if calibration else "no calibration applied",
                                   pl.PALETTE["black"])]:
        try:
            frac_pos, mean_pred = calibration_curve(y, scores, n_bins=10, strategy="uniform")
            ax.plot(mean_pred, frac_pos, marker="o", label=label, color=color)
        except ValueError:
            pass
    ax.plot([0, 1], [0, 1], linestyle="--", color=pl.PALETTE["sky_blue"], label="perfect calibration")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraction phishing")
    ax.legend(frameon=False)
    pl.save(fig, common.FIGURES_DIR / "fig4_reliability_test.png")


if __name__ == "__main__":
    fig2_pr_curves()
    fig3_confusion_three_outcome()
    fig4_reliability()
