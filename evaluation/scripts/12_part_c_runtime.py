"""Part C: runtime measurement.

Tries to start the real app server from the repository root and send 200 requests
built from development cases to POST /api/analyze on 127.0.0.1, recording
per-request latency (cold first request separate), median, 95th percentile,
failures and peak process memory. Labelled "end to end" in that case.

If the server cannot be started, falls back to measuring the same work
through the adapters directly in one process, labelled "in-process, not
end to end" as the spec requires.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import requests

import common

CASES = common.CACHE_DIR / "case_table.parquet"
SPLIT = common.CACHE_DIR / "split.parquet"
OUT_LAT = common.CACHE_DIR / "runtime_latencies.parquet"
OUT_TABLE = common.TABLES_DIR / "runtime.csv"
OUT_ENV = common.TABLES_DIR / "environment.csv"

N_REQUESTS = 200
HOST = "127.0.0.1"
PORT = 8123


def environment_table():
    import torch
    import transformers
    import sklearn
    rows = [
        {"key": "os", "value": platform.platform()},
        {"key": "python", "value": platform.python_version()},
        {"key": "cpu_model", "value": platform.processor() or "unknown"},
        {"key": "logical_cpus", "value": psutil.cpu_count(logical=True)},
        {"key": "physical_cpus", "value": psutil.cpu_count(logical=False)},
        {"key": "ram_gb", "value": round(psutil.virtual_memory().total / (1024 ** 3), 1)},
        {"key": "torch_version", "value": torch.__version__},
        {"key": "transformers_version", "value": transformers.__version__},
        {"key": "sklearn_version", "value": sklearn.__version__},
        {"key": "numpy_version", "value": np.__version__},
    ]
    pd.DataFrame(rows).to_csv(OUT_ENV, index=False)
    return rows


def try_start_server(phishlens_dir: Path):
    env = dict(**__import__("os").environ)
    env["USE_TORCH"] = "1"
    env["USE_TF"] = "0"
    env["USE_FLAX"] = "0"
    env["HF_HUB_OFFLINE"] = "1"
    env["PHISHLENS_HOME"] = str(phishlens_dir)
    py = str(phishlens_dir / ".venv" / "Scripts" / "python.exe")
    # The server log goes to a file. An undrained pipe fills up and blocks
    # the server once it has written enough log lines.
    log_path = Path(__file__).resolve().parent / "runtime_server.log"
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [py, "-m", "uvicorn", "phishlens.api:app", "--host", HOST, "--port", str(PORT),
         "--log-level", "warning"],
        cwd=str(phishlens_dir), env=env,
        stdout=log_file, stderr=subprocess.STDOUT, text=True,
    )
    url = f"http://{HOST}:{PORT}/api/health"
    for _ in range(60):
        if proc.poll() is not None:
            log_file.close()
            out = log_path.read_text(encoding="utf-8", errors="replace")
            return None, out
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return proc, None
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    proc.terminate()
    return None, "server did not become healthy within 60s"


def run_end_to_end(cases: pd.DataFrame):
    phishlens_dir = common.PHISHLENS_DIR
    proc, err = try_start_server(phishlens_dir)
    if proc is None:
        return None, err

    proc_ps = psutil.Process(proc.pid)
    latencies = []
    failures = 0
    peak_mem = 0
    sample = cases.sample(n=min(N_REQUESTS, len(cases)), random_state=common.SEED).reset_index(drop=True)
    try:
        for i, r in sample.iterrows():
            files = {}
            data = {"url": r["url"], "message_text": r["text"][:9999]}
            shot_path = common.WORK_DIR / r["screenshot_path"]
            fh = open(shot_path, "rb")
            files["screenshot"] = (r["id"] + ".jpg", fh, "image/jpeg")
            t0 = time.perf_counter()
            try:
                resp = requests.post(f"http://{HOST}:{PORT}/api/analyze", data=data, files=files, timeout=60)
                ok = resp.status_code == 200
            except requests.exceptions.RequestException:
                ok = False
            finally:
                fh.close()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if not ok:
                failures += 1
            latencies.append({"i": i, "id": r["id"], "latency_ms": dt_ms, "ok": ok})
            try:
                # On Windows the venv python.exe is a launcher that starts the
                # real interpreter as a child, so the children are summed too.
                procs = [proc_ps] + proc_ps.children(recursive=True)
                mem = sum(p.memory_info().rss for p in procs)
                peak_mem = max(peak_mem, mem)
            except psutil.Error:
                pass
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    return latencies, {"peak_memory_mb": round(peak_mem / 1e6, 1), "failures": failures}


def main() -> None:
    environment_table()
    cases = pd.read_parquet(CASES)
    split = pd.read_parquet(SPLIT)
    dev_ids = set(split[split.split == "dev"]["id"])
    dev_cases = cases[cases["id"].isin(dev_ids)].reset_index(drop=True)

    latencies, meta = run_end_to_end(dev_cases)
    mode = "end_to_end"
    if latencies is None:
        print("could not start the app server:", meta)
        print("falling back to in-process adapter timing")
        mode = "in_process_not_end_to_end"
        common.ensure_src_on_path()
        common.configure_hf_env()
        from phishlens.adapters.url_model import UrlModelAdapter
        from phishlens.adapters.text_model import TextModelAdapter
        from phishlens.adapters.image_model import ImageModelAdapter
        from PIL import Image

        url_a, text_a, img_a = UrlModelAdapter(), TextModelAdapter(), ImageModelAdapter()
        url_a.load(); text_a.load(); img_a.load()
        proc_ps = psutil.Process()
        sample = dev_cases.sample(n=min(N_REQUESTS, len(dev_cases)), random_state=common.SEED).reset_index(drop=True)
        latencies = []
        peak_mem = 0
        for i, r in sample.iterrows():
            t0 = time.perf_counter()
            url_a.predict(r["url"])
            text_a.predict(r["text"][:9999])
            img = Image.open(common.WORK_DIR / r["screenshot_path"]).convert("RGB")
            img_a.predict(img)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append({"i": i, "id": r["id"], "latency_ms": dt_ms, "ok": True})
            peak_mem = max(peak_mem, proc_ps.memory_info().rss)
        meta = {"peak_memory_mb": round(peak_mem / 1e6, 1), "failures": 0}

    lat_df = pd.DataFrame(latencies)
    lat_df.to_parquet(OUT_LAT, index=False)

    cold = lat_df.iloc[0]["latency_ms"]
    warm = lat_df.iloc[1:]["latency_ms"]
    summary = {
        "mode": mode,
        "n_requests": len(lat_df),
        "cold_first_request_ms": float(cold),
        "warm_median_ms": float(warm.median()),
        "warm_p95_ms": float(warm.quantile(0.95)),
        "warm_min_ms": float(warm.min()),
        "warm_max_ms": float(warm.max()),
        "failures": int(meta["failures"]),
        "peak_process_memory_mb": meta["peak_memory_mb"],
    }
    pd.DataFrame([summary]).to_csv(OUT_TABLE, index=False)
    print(json.dumps(summary, indent=2))
    print("wrote", OUT_LAT)
    print("wrote", OUT_TABLE)


if __name__ == "__main__":
    main()
