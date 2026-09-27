"""Figure 1: case counts by label at each filtering step, and split sizes."""
from __future__ import annotations

import pandas as pd

import common
import plotting as pl
import matplotlib.pyplot as plt
import numpy as np

STEPS_CSV = common.TABLES_DIR / "dataset_summary.csv"
SPLIT_MANIFEST = common.OUTPUTS_DIR / "split_manifest.json"


def main() -> None:
    steps = pd.read_csv(STEPS_CSV)
    import json
    manifest = json.load(open(SPLIT_MANIFEST))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    labels = [s.replace("_", " ") for s in steps["step"]]
    phishing = steps["phishing"].fillna(0).to_numpy()
    legit = steps["legitimate"].fillna(0).to_numpy()
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, phishing, w, label="phishing", color=pl.PALETTE["vermillion"])
    ax.bar(x + w / 2, legit, w, label="legitimate", color=pl.PALETTE["blue"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("case count")
    ax.legend(frameon=False)

    ax = axes[1]
    dev = manifest["dev"]
    test = manifest["test"]
    cats = ["development", "locked test"]
    p = [dev["phishing"], test["phishing"]]
    l = [dev["legitimate"], test["legitimate"]]
    x = np.arange(len(cats))
    ax.bar(x - w / 2, p, w, label="phishing", color=pl.PALETTE["vermillion"])
    ax.bar(x + w / 2, l, w, label="legitimate", color=pl.PALETTE["blue"])
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("case count")
    ax.legend(frameon=False)

    pl.save(fig, common.FIGURES_DIR / "fig1_case_counts_and_split.png")


if __name__ == "__main__":
    main()
