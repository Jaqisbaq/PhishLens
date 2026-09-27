"""Figure 5: ablation bar chart with intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
import plotting as pl
import matplotlib.pyplot as plt

ABLATION = common.TABLES_DIR / "ablation.csv"


def main() -> None:
    df = pd.read_csv(ABLATION)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    ap = df["average_precision"].to_numpy()
    lo = ap - df["ap_ci_low"].to_numpy()
    hi = df["ap_ci_high"].to_numpy() - ap
    ax.bar(x, ap, color=pl.PALETTE["blue"], yerr=[lo, hi], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df["branches_masked"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("average precision (test, group bootstrap 95% interval)")
    ax.set_ylim(0, 1.02)
    pl.save(fig, common.FIGURES_DIR / "fig5_ablation.png")


if __name__ == "__main__":
    main()
