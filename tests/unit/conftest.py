"""Fake adapters for unit tests. No torch, no transformers, no network."""
from __future__ import annotations

import time
from typing import Any, Optional

import pytest

from phishlens.adapters.base import BaseAdapter
from phishlens.schemas import BranchEvidence


class FakeAdapter(BaseAdapter):
    """An adapter whose probability, details and failure mode are set in the test."""

    def __init__(self, name: str, model_id: str = "fake/model", revision: str = "fakerev",
                 probability: float = 0.5, truncated: bool = False, details: Optional[dict] = None,
                 raise_on_predict: Optional[Exception] = None, latency_s: float = 0.0):
        self.name = name
        self.model_id = model_id
        self.revision = revision
        super().__init__()
        self.probability = probability
        self.truncated = truncated
        self.details = details or {}
        self.raise_on_predict = raise_on_predict
        self.latency_s = latency_s
        self.load_calls = 0
        self.predict_calls = 0

    def _load(self) -> None:
        self.load_calls += 1

    def _predict(self, value: Any) -> BranchEvidence:
        self.predict_calls += 1
        if self.latency_s:
            time.sleep(self.latency_s)
        if self.raise_on_predict is not None:
            raise self.raise_on_predict
        return BranchEvidence(
            branch=self.name,
            probability=self.probability,
            truncated=self.truncated,
            details=dict(self.details),
        )


@pytest.fixture
def fake_adapters():
    return {
        "url": FakeAdapter("url", model_id="fake/url-model", probability=0.2),
        "text": FakeAdapter("text", model_id="fake/text-model", probability=0.3),
        "image": FakeAdapter("image", model_id="fake/image-model", probability=0.1),
    }
