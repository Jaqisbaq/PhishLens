"""Figure 7: candidate comparison chart (average precision per model)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
import plotting as pl
import matplotlib.pyplot as plt

TABLE = common.TABLES_DIR / "candidate_comparison.csv"


def main() -> None:
    if not TABLE.exists():
        print("no candidate comparison table found, skipping figure 7")
        return
    df = pd.read_csv(TABLE)
    df = df[df["average_precision"].notna()].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [pl.PALETTE["blue"] if m == "url" else pl.PALETTE["vermillion"] for m in df["modality"]]
    y = np.arange(len(df))
    ax.barh(y, df["average_precision"], color=colors)
    labels = [
        f"{row.model[:55]}{'...' if len(row.model) > 55 else ''}"
        + (" [contaminated]" if bool(row.contaminated_on_this_sample) else "")
        for row in df.itertuples()
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("average precision")
    ax.set_xlim(0, 1.02)
    ax.invert_yaxis()
    pl.save(fig, common.FIGURES_DIR / "fig7_candidate_comparison.png")


if __name__ == "__main__":
    main()
