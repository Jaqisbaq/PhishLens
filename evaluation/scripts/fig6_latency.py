"""Figure 6: latency distribution from the Part C runtime measurement."""
from __future__ import annotations

import pandas as pd

import common
import plotting as pl
import matplotlib.pyplot as plt

LATENCY = common.CACHE_DIR / "runtime_latencies.parquet"


def main() -> None:
    if not LATENCY.exists():
        print("no runtime latency cache found, skipping figure 6 (Part C was not run)")
        return
    df = pd.read_parquet(LATENCY)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df["latency_ms"], bins=30, color=pl.PALETTE["blue"], edgecolor="white")
    ax.set_xlabel("per-request latency (ms)")
    ax.set_ylabel("request count")
    pl.save(fig, common.FIGURES_DIR / "fig6_latency_distribution.png")


if __name__ == "__main__":
    main()
