"""Capture interface screenshots against the running app with real models.

Starts the FastAPI app in a background thread (models load on first use, or
preloaded up front), drives it with Playwright, and saves PNGs plus a JSON
sidecar describing each shot to evidence/screenshots/.

If Playwright's browsers are not installed and cannot be installed without
network access, this prints a message and exits; the rest of the evidence
is unaffected.

Usage (Windows):
  .venv\\Scripts\\python.exe scripts\\capture_screenshots.py
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PORT = int(os.environ.get("PHISHLENS_SCREENSHOT_PORT", "8731"))
BASE_URL = "http://127.0.0.1:%d" % PORT
OUT_DIR = ROOT / "evidence" / "screenshots"


def start_server():
    import uvicorn

    from phishlens.api import app, get_orchestrator

    get_orchestrator().preload()

    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if getattr(server, "started", False):
            break
        time.sleep(0.1)
    return server


def wait_ready():
    import urllib.request

    for _ in range(100):
        try:
            with urllib.request.urlopen(BASE_URL + "/api/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:  # noqa: BLE001
            time.sleep(0.2)
    raise RuntimeError("server did not become ready")


def main() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # noqa: BLE001
        print("Playwright is not importable: %s" % exc)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    server = start_server()
    wait_ready()

    fixtures_dir = ROOT / "fixtures"
    with open(fixtures_dir / "cases.json", "r", encoding="utf-8") as handle:
        cases = json.load(handle)

    shots = []
    try:
        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch()
            except Exception as exc:  # noqa: BLE001
                print(
                    "Playwright's Chromium browser is not installed and could not be "
                    "launched (%s). Run '.venv\\Scripts\\python.exe -m playwright install "
                    "chromium' with network access, then re-run this script." % exc
                )
                return

            page = browser.new_page(viewport={"width": 1000, "height": 900})

            # 1. Empty form
            page.goto(BASE_URL + "/")
            page.wait_for_selector("#analyze-form")
            path = OUT_DIR / "01_empty_form.png"
            page.screenshot(path=str(path), full_page=True)
            shots.append({"file": path.name, "description": "Empty form on first load."})

            # 2. Complete three-input harmless case (yields "Limited indicators"
            # with the fitted fusion parameters).
            page.fill("#url", cases["urls"]["benign_plain"])
            page.fill("#message_text", cases["messages"]["benign"])
            page.set_input_files("#screenshot", str(fixtures_dir / cases["images"]["news_article"]))
            page.click("#submit-btn")
            page.wait_for_selector("#result-panel:not(.hidden)", timeout=30000)
            path = OUT_DIR / "02_complete_harmless_case.png"
            page.screenshot(path=str(path), full_page=True)
            shots.append({
                "file": path.name,
                "description": "Result panel for a complete harmless case (url + text + "
                               "screenshot); outcome is 'Limited indicators' with the fitted "
                               "fusion parameters.",
            })

            # 2b. Complete three-input suspicious case (yields "High concern"
            # with the fitted fusion parameters). All inputs are synthetic
            # fixtures: a TEST-NET-1 (192.0.2.0/24, RFC 5737) IP-literal URL
            # shaped like a credential-harvesting link, a synthetic urgent
            # message, and the fixture login-page screenshot.
            page.goto(BASE_URL + "/")
            page.wait_for_selector("#analyze-form")
            page.fill("#url", cases["urls"]["suspicious_shape"])
            page.fill("#message_text", cases["messages"]["urgent_style"])
            page.set_input_files("#screenshot", str(fixtures_dir / cases["images"]["login_page"]))
            page.click("#submit-btn")
            page.wait_for_selector("#result-panel:not(.hidden)", timeout=30000)
            outcome_text = page.text_content("#outcome-badge") or ""
            if "High concern" in outcome_text:
                path = OUT_DIR / "06_high_concern_case.png"
                page.screenshot(path=str(path), full_page=True)
                shots.append({
                    "file": path.name,
                    "description": "Result panel for a complete case built from a suspicious-"
                                   "shaped synthetic URL, an urgent synthetic message, and the "
                                   "fixture login-page screenshot; outcome is 'High concern' "
                                   "with the fitted fusion parameters.",
                })
            else:
                print(
                    "Skipped the 'High concern' screenshot: this fixture combination "
                    "produced outcome '%s' instead." % outcome_text
                )

            # 3. Partial case (url only)
            page.goto(BASE_URL + "/")
            page.wait_for_selector("#analyze-form")
            page.fill("#url", cases["urls"]["benign_org"])
            page.click("#submit-btn")
            page.wait_for_selector("#result-panel:not(.hidden)", timeout=30000)
            path = OUT_DIR / "03_partial_case_url_only.png"
            page.screenshot(path=str(path), full_page=True)
            shots.append({"file": path.name, "description": "Result panel for a url-only (partial) case."})

            # 4. Validation error
            page.goto(BASE_URL + "/")
            page.wait_for_selector("#analyze-form")
            page.fill("#url", "not-a-valid-url")
            page.click("#submit-btn")
            page.wait_for_selector("#form-errors:not(.hidden)", timeout=10000)
            path = OUT_DIR / "04_validation_error.png"
            page.screenshot(path=str(path), full_page=True)
            shots.append({"file": path.name, "description": "Inline validation error for a malformed URL."})

            # 5. Review needed case, only if this fixture actually produces it.
            page.goto(BASE_URL + "/")
            page.wait_for_selector("#analyze-form")
            page.fill("#url", cases["urls"]["suspicious_shape"])
            page.fill("#message_text", cases["messages"]["benign"])
            page.click("#submit-btn")
            page.wait_for_selector("#result-panel:not(.hidden)", timeout=30000)
            outcome_text = page.text_content("#outcome-badge") or ""
            if "Review needed" in outcome_text:
                path = OUT_DIR / "05_review_needed_case.png"
                page.screenshot(path=str(path), full_page=True)
                shots.append({
                    "file": path.name,
                    "description": "A case that produced 'Review needed' naturally from mixed evidence.",
                })
            else:
                print(
                    "Skipped the 'Review needed' screenshot: this fixture combination "
                    "produced outcome '%s' instead." % outcome_text
                )

            browser.close()
    finally:
        server.should_exit = True

    with open(OUT_DIR / "screenshots.json", "w", encoding="utf-8") as handle:
        json.dump({"captured": shots, "base_url": BASE_URL}, handle, indent=2)
    print("Captured %d screenshots into %s" % (len(shots), OUT_DIR))


if __name__ == "__main__":
    main()
