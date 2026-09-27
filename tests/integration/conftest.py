"""Shared fixtures for the slow integration tests (real models, real fixtures)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = ROOT / "fixtures"


@pytest.fixture(scope="session")
def cases():
    with open(FIXTURES_DIR / "cases.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def fixture_image_paths():
    return {
        name: FIXTURES_DIR / rel
        for name, rel in json.load(open(FIXTURES_DIR / "cases.json", encoding="utf-8"))["images"].items()
    }


@pytest.fixture(scope="session")
def orchestrator():
    from phishlens.orchestrator import Orchestrator

    orch = Orchestrator()
    orch.preload()
    return orch
