"""Run the PhishLens server on 127.0.0.1 only.

Usage (Windows):
  .venv\\Scripts\\python.exe scripts\\run_server.py [--port 8000]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preload", action="store_true",
                         help="Load all three models before accepting requests.")
    args = parser.parse_args()

    import uvicorn

    from phishlens import config
    from phishlens.api import app, get_orchestrator

    if args.preload:
        get_orchestrator().preload()

    uvicorn.run(app, host=config.HOST, port=args.port)


if __name__ == "__main__":
    main()
