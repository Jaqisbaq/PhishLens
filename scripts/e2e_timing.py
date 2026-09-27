"""Manual end-to-end timing: load all three real models once, then analyze
one fixture case (url + text + image together) in the same process.

Usage (Windows):
  .venv\\Scripts\\python.exe scripts\\e2e_timing.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main() -> None:
    from phishlens.orchestrator import Orchestrator
    from phishlens.validation import validate_case

    fixtures_dir = ROOT / "fixtures"
    with open(fixtures_dir / "cases.json", "r", encoding="utf-8") as handle:
        cases = json.load(handle)

    print("Loading all three real models (once)...")
    t0 = time.perf_counter()
    orch = Orchestrator()
    orch.preload()
    load_ms = (time.perf_counter() - t0) * 1000.0
    print("Model load time: %.1f ms" % load_ms)

    screenshot_path = fixtures_dir / cases["images"]["login_page"]
    with open(screenshot_path, "rb") as handle:
        screenshot_bytes = handle.read()

    case = validate_case(
        url=cases["urls"]["suspicious_shape"],
        message_text=cases["messages"]["urgent_style"],
        screenshot_bytes=screenshot_bytes,
    )

    print("Analyzing one case (url + text + image)...")
    t1 = time.perf_counter()
    result = orch.analyze(case)
    analyze_ms = (time.perf_counter() - t1) * 1000.0

    print("Analysis latency (in-process, models already loaded): %.2f ms" % analyze_ms)
    print("Orchestrator-reported total_latency_ms: %s" % result.total_latency_ms)
    print("Outcome: %s" % result.outcome)
    print("Fused score: %s" % result.fused_score)
    for b in result.branches:
        print(
            "  branch=%s status=%s probability=%s latency_ms=%s model=%s@%s"
            % (b.branch, b.status, b.probability, b.latency_ms, b.model_id, (b.revision or "")[:8])
        )
    print()
    print("TOTAL (load once + analyze one case): %.1f ms" % (load_ms + analyze_ms))


if __name__ == "__main__":
    main()
