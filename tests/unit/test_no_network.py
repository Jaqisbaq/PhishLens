"""Hard safety requirement: no DNS, no HTTP, ever, during analysis of a case.

socket.socket.connect and socket.getaddrinfo are monkeypatched to raise. If
anything in validation, the URL adapter's feature extraction, the
orchestrator or the API tried to resolve or open the submitted URL, this
test would fail with the AssertionError raised from the patched functions.
"""
from __future__ import annotations

import socket

import pytest

from phishlens.fusion import FusionConfig
from phishlens.orchestrator import Orchestrator
from phishlens.schemas import CaseInput
from phishlens.validation import validate_case


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def boom_connect(*args, **kwargs):
        raise AssertionError("socket.socket.connect was called: network access attempted")

    def boom_getaddrinfo(*args, **kwargs):
        raise AssertionError("socket.getaddrinfo was called: network access attempted")

    monkeypatch.setattr(socket.socket, "connect", boom_connect)
    monkeypatch.setattr(socket, "getaddrinfo", boom_getaddrinfo)
    yield


def test_validation_does_not_touch_the_network():
    case = validate_case(url="http://192.0.2.77/definitely-not-fetched")
    assert case.url == "http://192.0.2.77/definitely-not-fetched"


def test_orchestrator_does_not_touch_the_network(fake_adapters):
    orch = Orchestrator(adapters=fake_adapters)
    orch.fusion_config = FusionConfig().validate()
    case = CaseInput(url="https://example.com/should-not-be-fetched", message_text="hi")
    result = orch.analyze(case)
    assert result.outcome
