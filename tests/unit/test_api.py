"""API: happy path, validation errors, health, and no-network proof."""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from phishlens.api import app, set_orchestrator
from phishlens.fusion import FusionConfig
from phishlens.orchestrator import Orchestrator


@pytest.fixture
def client(fake_adapters):
    orch = Orchestrator(adapters=fake_adapters)
    orch.fusion_config = FusionConfig().validate()
    set_orchestrator(orch)
    return TestClient(app)


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_index_page_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "PhishLens" in resp.text


def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["branches"]) == {"url", "text", "image"}
    assert "fusion" in data


def test_analyze_happy_path_all_three_inputs(client):
    resp = client.post(
        "/api/analyze",
        data={"url": "https://example.com/login", "message_text": "hello there"},
        files={"screenshot": ("shot.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["outcome"] in (
        "High concern", "Review needed", "Limited indicators", "Analysis unavailable"
    )
    assert len(data["branches"]) == 3
    assert data["caveat"]


def test_analyze_url_only(client):
    resp = client.post("/api/analyze", data={"url": "https://example.com/"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["inputs_provided"] == {"url": True, "text": False, "image": False}


def test_analyze_no_input_gives_422(client):
    resp = client.post("/api/analyze", data={})
    assert resp.status_code == 422
    assert resp.json()["errors"]


def test_analyze_bad_url_gives_422_with_message(client):
    resp = client.post("/api/analyze", data={"url": "not-a-url"})
    assert resp.status_code == 422
    errors = resp.json()["errors"]
    assert any(e["field"] == "url" for e in errors)


def test_analyze_bad_image_gives_422(client):
    resp = client.post(
        "/api/analyze",
        data={},
        files={"screenshot": ("bad.png", b"not an image", "image/png")},
    )
    assert resp.status_code == 422
    errors = resp.json()["errors"]
    assert any(e["field"] == "screenshot" for e in errors)


# The authoritative no-network proof (monkeypatching socket.socket.connect and
# socket.getaddrinfo around the analysis call, per the spec) is in
# test_no_network.py. It runs the orchestrator directly rather than through
# TestClient, because TestClient's own async transport uses a loopback
# socketpair for its event loop on Windows, which would make this test fail
# for a reason unrelated to the application code.
