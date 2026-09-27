"""Integration tests: the three real pretrained models against the fixtures.

Marked slow. Run with: pytest tests/integration -m slow
These load real weights from the local Hugging Face cache (HF_HUB_OFFLINE=1)
and take real inference time; they are not run on every unit test pass.
"""
from __future__ import annotations

import os

import pytest
from PIL import Image

os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

pytestmark = pytest.mark.slow


def test_url_adapter_predict_batch_real_model(orchestrator, cases):
    adapter = orchestrator.adapters["url"]
    urls = list(cases["urls"].values())
    probs = adapter.predict_batch(urls)
    assert len(probs) == len(urls)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_text_adapter_predict_batch_real_model(orchestrator, cases):
    adapter = orchestrator.adapters["text"]
    messages = list(cases["messages"].values())
    probs = adapter.predict_batch(messages)
    assert len(probs) == len(messages)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_image_adapter_embed_and_predict_real_model(orchestrator, fixture_image_paths):
    adapter = orchestrator.adapters["image"]
    images = [Image.open(p).convert("RGB") for p in fixture_image_paths.values()]
    embeddings = adapter.embed_images(images)
    assert embeddings.shape == (len(images), 512)
    probs = adapter.predict_batch(images)
    assert len(probs) == len(images)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_end_to_end_case_url_text_image(orchestrator, cases, fixture_image_paths):
    from phishlens.validation import validate_case

    with open(fixture_image_paths["login_page"], "rb") as handle:
        screenshot_bytes = handle.read()

    case = validate_case(
        url=cases["urls"]["suspicious_shape"],
        message_text=cases["messages"]["urgent_style"],
        screenshot_bytes=screenshot_bytes,
    )
    result = orchestrator.analyze(case)
    assert len(result.branches) == 3
    assert all(b.status == "ok" for b in result.branches)
    assert result.outcome in (
        "High concern", "Review needed", "Limited indicators", "Analysis unavailable"
    )
    assert result.fused_score is not None
    assert result.explanations


def test_end_to_end_benign_case(orchestrator, cases, fixture_image_paths):
    from phishlens.validation import validate_case

    with open(fixture_image_paths["news_article"], "rb") as handle:
        screenshot_bytes = handle.read()

    case = validate_case(
        url=cases["urls"]["benign_plain"],
        message_text=cases["messages"]["benign"],
        screenshot_bytes=screenshot_bytes,
    )
    result = orchestrator.analyze(case)
    assert all(b.status == "ok" for b in result.branches)
    assert result.fused_score is not None


def test_url_model_truncation_flag_on_long_url(orchestrator):
    # A repeated single character tokenizes into very few tokens with this
    # model's tokenizer (long merges), so the path needs varied characters
    # to actually exceed the 64 token limit.
    adapter = orchestrator.adapters["url"]
    import random

    rng = random.Random(1)
    segment = "".join(rng.choice("abcdefghijklmnop-/.0123456789") for _ in range(400))
    long_url = "https://example.com/" + segment
    from phishlens.schemas import BranchEvidence

    evidence = adapter.predict(long_url)
    assert isinstance(evidence, BranchEvidence)
    assert evidence.details["token_count"] > adapter.max_tokens
    assert evidence.truncated is True


def test_text_model_token_count_detail(orchestrator, cases):
    adapter = orchestrator.adapters["text"]
    evidence = adapter.predict(cases["messages"]["benign"])
    assert "token_count" in evidence.details
    assert evidence.details["token_count"] > 0
