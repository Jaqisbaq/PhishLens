"""Central configuration: model ids, pinned revisions, paths and limits.

All paths are resolved relative to the project folder, never hard coded.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"


def project_root() -> Path:
    """Project folder. Override with the PHISHLENS_HOME environment variable."""
    override = os.environ.get("PHISHLENS_HOME")
    if override:
        return Path(override)
    # src/phishlens/config.py -> project root is two levels above the package
    return PACKAGE_DIR.parent.parent


def models_dir() -> Path:
    override = os.environ.get("PHISHLENS_MODELS_DIR")
    if override:
        return Path(override)
    return project_root() / "models"


def fusion_path() -> Path:
    return models_dir() / "fusion.json"


def image_head_path() -> Path:
    return models_dir() / "image_head.json"


def manifest_path() -> Path:
    return models_dir() / "manifest.json"


# --------------------------------------------------------------------------
# Models (pinned revisions)
# --------------------------------------------------------------------------
TEXT_MODEL_ID = "ealvaradob/bert-finetuned-phishing"
TEXT_MODEL_REVISION = "fa8fb73a007174c410ab7160d4e4c6e6b8d998d4"
TEXT_MAX_TOKENS = 512
TEXT_PHISHING_INDEX = 1  # config labels: 0 benign, 1 phishing

URL_MODEL_ID = "CrabInHoney/urlbert-tiny-v4-phishing-classifier"
URL_MODEL_REVISION = "fd962a5cd04e20ceed46aa8a6e31a2b40d4c8916"
URL_MAX_TOKENS = 64
# The config only has generic LABEL_0 / LABEL_1. The mapping below comes from
# the model card (0 good, 1 phishing) and is hard coded on purpose.
URL_PHISHING_INDEX = 1

IMAGE_MODEL_ID = "openai/clip-vit-base-patch32"
IMAGE_MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
IMAGE_EMBEDDING_DIM = 512

# Zero-shot prompt set for the image branch (used when no fitted head exists).
SUSPICIOUS_PROMPTS = [
    "a screenshot of a login page asking for a password",
    "a screenshot of a web page asking for credit card details",
    "a screenshot of a message asking to confirm account or card details",
    "a screenshot of a prize notice or an urgent account warning",
]
ORDINARY_PROMPTS = [
    "a screenshot of a news article or blog post",
    "a screenshot of a documentation page",
    "a screenshot of a table of information",
    "a screenshot of an ordinary chat message",
]

# --------------------------------------------------------------------------
# Input limits
# --------------------------------------------------------------------------
MAX_URL_LENGTH = 2048
MAX_TEXT_LENGTH = 10_000
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000  # guard against decompression bombs
MAX_IMAGE_SIDE = 10_000
ALLOWED_IMAGE_FORMATS = ("PNG", "JPEG")
ALLOWED_URL_SCHEMES = ("http", "https")

# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------
HOST = "127.0.0.1"  # loopback only, never 0.0.0.0
PORT = 8000
RESULT_CACHE_SIZE = 50  # results kept in memory only, for the export endpoint

# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
LOAD_ATTEMPTS = 3
LOAD_RETRY_WAIT_S = 0.5

BRANCHES = ("url", "text", "image")


def configure_environment() -> None:
    """Set environment variables that must be in place before transformers is imported.

    USE_TORCH / USE_TF / USE_FLAX stop transformers from importing TensorFlow
    and JAX when they happen to be installed. HF_HUB_OFFLINE keeps inference
    fully local. To download the models the first time, set HF_HUB_OFFLINE=0
    in the shell before starting (see scripts/download_models.py).
    """
    os.environ.setdefault("USE_TORCH", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", os.environ["HF_HUB_OFFLINE"])
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
