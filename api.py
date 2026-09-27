"""FastAPI application: GET /, POST /api/analyze, GET /api/health.

Binds to 127.0.0.1 only (see scripts/run_server.py). Submitted content is
processed in memory and never written to disk. The submitted URL is only
ever passed to the model adapter as a string; nothing here resolves it.
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .orchestrator import Orchestrator
from .validation import ValidationError, validate_case

app = FastAPI(title="PhishLens", description="Local phishing triage, no outbound requests.")

_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


def set_orchestrator(orchestrator: Orchestrator) -> None:
    """Used by tests to inject an orchestrator built from fake adapters."""
    global _orchestrator
    _orchestrator = orchestrator


if config.STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")


@app.get("/")
def index():
    index_path = config.STATIC_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse({"error": "interface not found"}, status_code=404)
    return FileResponse(str(index_path))


@app.get("/api/health")
def health():
    return get_orchestrator().health()


@app.post("/api/analyze")
async def analyze(
    url: Optional[str] = Form(default=None),
    message_text: Optional[str] = Form(default=None),
    screenshot: Optional[UploadFile] = File(default=None),
):
    screenshot_bytes = None
    if screenshot is not None and screenshot.filename:
        screenshot_bytes = await screenshot.read()

    try:
        case = validate_case(
            url=url, message_text=message_text, screenshot_bytes=screenshot_bytes
        )
    except ValidationError as exc:
        return JSONResponse({"errors": exc.errors}, status_code=422)

    result = get_orchestrator().analyze(case)
    return JSONResponse(result.model_dump())
