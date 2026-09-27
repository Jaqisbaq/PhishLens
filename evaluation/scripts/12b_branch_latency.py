"""Part C supplement: model load time and latency per branch.

Runs the same 200 development cases as 12_part_c_runtime.py, in one process
through the application's orchestrator, and records the latency that each
branch reports for each case. The end to end figures in runtime.csv come
from the API run; this script only explains how that time is divided.

Outputs:
  outputs/tables/runtime_branches.csv
  outputs/cache/runtime_branch_latencies.parquet
"""
from __future__ import annotations

import json
import time

import pandas as pd

import common

N_REQUESTS = 200
CASES = common.CACHE_DIR / "case_table.parquet" if hasattr(common, "CACHE_DIR") else None


def main() -> None:
    common.ensure_src_on_path()
    common.configure_hf_env()
    from phishlens.orchestrator import Orchestrator
    from phishlens.validation import validate_case

    out_dir = common.PHISHLENS_DIR / "evaluation" / "outputs"
    cases = pd.read_parquet(out_dir / "cache" / "case_table.parquet")
    split = pd.read_parquet(out_dir / "cache" / "split.parquet")
    dev_ids = set(split[split.split == "dev"]["id"])
    dev = cases[cases["id"].isin(dev_ids)].reset_index(drop=True)
    sample = dev.sample(n=min(N_REQUESTS, len(dev)),
                        random_state=common.SEED).reset_index(drop=True)

    t0 = time.perf_counter()
    orch = Orchestrator()
    orch.preload()
    load_ms = (time.perf_counter() - t0) * 1000.0

    rows = []
    for i, r in sample.iterrows():
        with open(common.WORK_DIR / r["screenshot_path"], "rb") as fh:
            shot = fh.read()
        case = validate_case(url=r["url"], message_text=r["text"][:9999],
                             screenshot_bytes=shot)
        t1 = time.perf_counter()
        result = orch.analyze(case)
        total_ms = (time.perf_counter() - t1) * 1000.0
        row = {"i": i, "id": r["id"], "total_ms": total_ms}
        for b in result.branches:
            row[b.branch + "_ms"] = b.latency_ms
            row[b.branch + "_status"] = b.status
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "cache" / "runtime_branch_latencies.parquet", index=False)

    warm = df.iloc[1:]
    summary = {
        "mode": "in_process_orchestrator",
        "n_cases": int(len(df)),
        "model_load_ms": round(load_ms, 1),
        "warm_median_total_ms": round(float(warm["total_ms"].median()), 1),
    }
    for col in [c for c in df.columns if c.endswith("_ms") and c != "total_ms"]:
        summary["warm_median_" + col] = round(float(warm[col].median()), 1)
    branch_cols = [c for c in df.columns if c.endswith("_ms") and c != "total_ms"]
    branch_sum = warm[branch_cols].sum(axis=1)
    for col in branch_cols:
        share = (warm[col] / branch_sum).median() * 100.0
        summary["median_share_pct_" + col[:-3]] = round(float(share), 1)
    statuses = [c for c in df.columns if c.endswith("_status")]
    summary["branches_not_ok"] = int((df[statuses] != "ok").sum().sum())

    pd.DataFrame([summary]).to_csv(out_dir / "tables" / "runtime_branches.csv",
                                   index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
