"""Step 2: download the needed files of every candidate at the pinned revision.

Only inference files are fetched. Skipped on purpose:
  - TensorFlow / Flax weights (not used)
  - training_args.bin and model.pkl (pickle files that are not needed for inference)
  - notebooks
  - pytorch_model.bin when a safetensors file exists in the same repo

Reads metadata.json (from 01_metadata.py), writes downloads.json.

Run:  python 02_download.py
"""
import json
import os
import sys
import time

from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))

SKIP_ALWAYS = (
    "tf_model.h5",
    "flax_model.msgpack",
    "training_args.bin",
    "model.pkl",
    ".gitattributes",
    "gitattributes",
)


def wanted_files(files):
    names = [f["name"] for f in files]
    has_safetensors = any(n.endswith(".safetensors") for n in names)
    keep = []
    for n in names:
        if n in SKIP_ALWAYS or n.endswith(".ipynb"):
            continue
        if n == "pytorch_model.bin" and has_safetensors:
            continue
        keep.append(n)
    return keep, has_safetensors


def main():
    with open(os.path.join(HERE, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    only = sys.argv[1:]
    out_path = os.path.join(HERE, "downloads.json")
    out = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            out = json.load(f)
    for repo, rec in meta.items():
        if only and repo not in only:
            continue
        if rec.get("metadata_status") != "OK":
            continue
        keep, has_st = wanted_files(rec["files"])
        t0 = time.perf_counter()
        try:
            path = snapshot_download(
                repo, revision=rec["revision"], allow_patterns=keep
            )
            dt = time.perf_counter() - t0
            size = 0
            present = []
            for root, _, fs in os.walk(path):
                for fn in fs:
                    fp = os.path.join(root, fn)
                    size += os.path.getsize(fp)
                    present.append(os.path.relpath(fp, path).replace("\\", "/"))
            out[repo] = {
                "status": "OK",
                "revision": rec["revision"],
                "download_seconds": round(dt, 2),
                "downloaded_bytes_on_disk": size,
                "files_present": sorted(present),
                "weights_format": "safetensors" if has_st else "pytorch_bin_or_other",
            }
        except Exception as e:  # noqa: BLE001
            out[repo] = {"status": "ERROR", "error": repr(e)}
        print(repo, out[repo].get("status"), out[repo].get("download_seconds"),
              out[repo].get("downloaded_bytes_on_disk"), flush=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
