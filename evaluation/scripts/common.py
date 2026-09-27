"""Shared paths and helpers for the PhishLens evaluation scripts.

All paths are resolved relative to this file so nothing absolute or
machine-specific is baked into any output.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

csv.field_size_limit(10**9)

SCRIPTS_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPTS_DIR.parent
PHISHLENS_DIR = EVAL_DIR.parent
WORK_DIR = PHISHLENS_DIR.parent
DATA_DIR = WORK_DIR / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_LOGS = DATA_DIR / "_logs"

OUTPUTS_DIR = EVAL_DIR / "outputs"
CACHE_DIR = OUTPUTS_DIR / "cache"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
FITTED_DIR = OUTPUTS_DIR / "fitted"

for d in (OUTPUTS_DIR, CACHE_DIR, TABLES_DIR, FIGURES_DIR, FITTED_DIR):
    d.mkdir(parents=True, exist_ok=True)

SRC_DIR = PHISHLENS_DIR / "src"

SEED = 2026


def configure_hf_env() -> None:
    os.environ.setdefault("USE_TORCH", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def ensure_src_on_path() -> None:
    import sys
    p = str(SRC_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
