"""Writes outputs/run_manifest.json: date, seeds, model revisions, package
versions, split hash, inference path used, and leakage-removal counts.
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone

import pandas as pd

import common
common.ensure_src_on_path()

OUT = common.OUTPUTS_DIR / "run_manifest.json"


def main() -> None:
    import torch
    import transformers
    import sklearn
    import numpy as np
    from phishlens import config as pconfig

    split_manifest = json.load(open(common.OUTPUTS_DIR / "split_manifest.json"))
    steps = pd.read_csv(common.TABLES_DIR / "dataset_summary.csv")

    inference_path = "adapters"
    try:
        from phishlens.adapters.url_model import UrlModelAdapter  # noqa: F401
        from phishlens.adapters.text_model import TextModelAdapter  # noqa: F401
        from phishlens.adapters.image_model import ImageModelAdapter  # noqa: F401
        from phishlens import fusion  # noqa: F401
        inference_path = "adapters (src/phishlens/adapters and phishlens.fusion " \
                          "imported cleanly in the venv and were used directly)"
    except Exception as exc:  # noqa: BLE001
        inference_path = "direct-load per docs/model_selection/04_bench.py (adapters import failed: %r)" % exc

    manifest = {
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": common.SEED,
        "python": platform.python_version(),
        "package_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "model_revisions": {
            "text": {"model_id": pconfig.TEXT_MODEL_ID, "revision": pconfig.TEXT_MODEL_REVISION},
            "url": {"model_id": pconfig.URL_MODEL_ID, "revision": pconfig.URL_MODEL_REVISION},
            "image": {"model_id": pconfig.IMAGE_MODEL_ID, "revision": pconfig.IMAGE_MODEL_REVISION},
        },
        "inference_path": inference_path,
        "split": {
            "manifest_sha256_of_body": split_manifest["sha256_of_manifest_body"],
            "seed": split_manifest["seed"],
            "dev": split_manifest["dev"],
            "test": split_manifest["test"],
        },
        "leakage_removal_counts_part_a": steps.to_dict(orient="records"),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps(manifest, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
