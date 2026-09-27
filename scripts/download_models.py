"""Download the three pinned models into the local Hugging Face cache.

This must be run once, with network access, before the server can work
offline. It downloads the exact pinned revisions listed in
src/phishlens/config.py (also mirrored in models/manifest.json), retries
each download once on failure (Windows can raise a symlink privilege
OSError on the first attempt and succeed on the second), and then loads
each model from the cache to verify it actually works before finishing.

Usage (Windows):
  .venv\\Scripts\\python.exe scripts\\download_models.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Downloading requires HF_HUB_OFFLINE=0. Set this before importing
# transformers/huggingface_hub or anything that imports phishlens.config,
# which defaults HF_HUB_OFFLINE to "1" for normal (offline) operation.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from phishlens import config  # noqa: E402

ATTEMPTS = 2
RETRY_WAIT_S = 3.0


def _download_with_retry(label: str, fn, *args, **kwargs):
    last_exc = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            print(f"[{label}] download attempt {attempt}/{ATTEMPTS} ...")
            result = fn(*args, **kwargs)
            print(f"[{label}] downloaded OK.")
            return result
        except OSError as exc:
            last_exc = exc
            print(f"[{label}] attempt {attempt} failed: {exc}")
            if attempt < ATTEMPTS:
                print(f"[{label}] retrying in {RETRY_WAIT_S:.0f}s "
                      f"(this is often a one-off Windows symlink privilege error) ...")
                time.sleep(RETRY_WAIT_S)
    raise RuntimeError(f"[{label}] failed after {ATTEMPTS} attempts") from last_exc


def download_url_model():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    def _fetch():
        AutoTokenizer.from_pretrained(config.URL_MODEL_ID, revision=config.URL_MODEL_REVISION)
        AutoModelForSequenceClassification.from_pretrained(
            config.URL_MODEL_ID, revision=config.URL_MODEL_REVISION
        )

    _download_with_retry("url", _fetch)


def download_text_model():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    def _fetch():
        AutoTokenizer.from_pretrained(config.TEXT_MODEL_ID, revision=config.TEXT_MODEL_REVISION)
        AutoModelForSequenceClassification.from_pretrained(
            config.TEXT_MODEL_ID, revision=config.TEXT_MODEL_REVISION
        )

    _download_with_retry("text", _fetch)


def download_image_model():
    from transformers import CLIPModel, CLIPProcessor

    def _fetch():
        CLIPProcessor.from_pretrained(config.IMAGE_MODEL_ID, revision=config.IMAGE_MODEL_REVISION)
        CLIPModel.from_pretrained(config.IMAGE_MODEL_ID, revision=config.IMAGE_MODEL_REVISION)

    _download_with_retry("image", _fetch)


def verify_offline_load() -> None:
    """Switch to offline mode and load each model from the cache only.

    This proves the download actually populated the cache with the pinned
    revisions, and that the app's normal offline startup path will work.
    """
    print("\nVerifying offline load from cache (HF_HUB_OFFLINE=1) ...")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        CLIPModel,
        CLIPProcessor,
    )

    AutoTokenizer.from_pretrained(config.URL_MODEL_ID, revision=config.URL_MODEL_REVISION)
    AutoModelForSequenceClassification.from_pretrained(
        config.URL_MODEL_ID, revision=config.URL_MODEL_REVISION
    )
    print("[url] loads offline OK.")

    AutoTokenizer.from_pretrained(config.TEXT_MODEL_ID, revision=config.TEXT_MODEL_REVISION)
    AutoModelForSequenceClassification.from_pretrained(
        config.TEXT_MODEL_ID, revision=config.TEXT_MODEL_REVISION
    )
    print("[text] loads offline OK.")

    CLIPProcessor.from_pretrained(config.IMAGE_MODEL_ID, revision=config.IMAGE_MODEL_REVISION)
    CLIPModel.from_pretrained(config.IMAGE_MODEL_ID, revision=config.IMAGE_MODEL_REVISION)
    print("[image] loads offline OK.")


def main() -> None:
    print("Downloading pinned models into the local Hugging Face cache.")
    print("This is roughly 2.3 GB total and only needs to happen once.\n")
    print(f"  url   {config.URL_MODEL_ID}   @ {config.URL_MODEL_REVISION}")
    print(f"  text  {config.TEXT_MODEL_ID}  @ {config.TEXT_MODEL_REVISION}")
    print(f"  image {config.IMAGE_MODEL_ID} @ {config.IMAGE_MODEL_REVISION}\n")

    download_url_model()
    download_text_model()
    download_image_model()
    verify_offline_load()

    print("\nAll three models downloaded and verified. You can now run the app "
          "offline with run.bat.")


if __name__ == "__main__":
    main()
