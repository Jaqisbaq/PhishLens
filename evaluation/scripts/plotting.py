"""Shared matplotlib setup: colour-blind-safe palette, 200 dpi PNG output."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colour-blind-safe palette.
PALETTE = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
    "grey": "#999999",
}

BRANCH_COLORS = {
    "url": PALETTE["blue"],
    "text": PALETTE["vermillion"],
    "image": PALETTE["bluish_green"],
    "mean": PALETTE["orange"],
    "max": PALETTE["reddish_purple"],
    "learned": PALETTE["black"],
}


def save(fig, path, dpi=200):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print("wrote", path)
