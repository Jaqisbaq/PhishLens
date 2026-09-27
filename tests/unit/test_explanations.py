"""Explanations are built only from evidence actually present."""
from __future__ import annotations

from phishlens.explanations import build_explanations
from phishlens.fusion import FusionConfig
from phishlens.orchestrator import Orchestrator
from phishlens.schemas import CaseInput


def test_explanation_mentions_missing_branch_by_name(fake_adapters):
    orch = Orchestrator(adapters=fake_adapters)
    orch.fusion_config = FusionConfig().validate()
    result = orch.analyze(CaseInput(url="https://example.com/"))
    joined = " ".join(result.explanations)
    assert "message text" in joined
    assert "screenshot" in joined


def test_explanation_never_invents_a_value_not_in_evidence(fake_adapters):
    fake_adapters["url"].details = {}
    orch = Orchestrator(adapters=fake_adapters)
    orch.fusion_config = FusionConfig().validate()
    result = orch.analyze(
        CaseInput(url="https://example.com/", message_text="hi", screenshot=object())
    )
    joined = " ".join(result.explanations)
    # No URL structural feature sentence should appear when details are empty.
    assert "Structural features of the URL" not in joined


def test_explanation_reports_truncation_only_when_flagged(fake_adapters):
    fake_adapters["text"].truncated = True
    fake_adapters["text"].details = {"token_count": 900, "max_tokens": 512}
    orch = Orchestrator(adapters=fake_adapters)
    orch.fusion_config = FusionConfig().validate()
    result = orch.analyze(
        CaseInput(url="https://example.com/", message_text="hi", screenshot=object())
    )
    joined = " ".join(result.explanations)
    assert "900 tokens" in joined
    assert "limit 512" in joined


def test_explanation_includes_provisional_notice_when_provisional(fake_adapters):
    orch = Orchestrator(adapters=fake_adapters)
    orch.fusion_config = FusionConfig().validate()  # default is provisional
    result = orch.analyze(CaseInput(url="https://example.com/"))
    assert result.provisional_settings is True
    assert any("Provisional settings" in s for s in result.explanations)


def test_explanation_disagreement_names_both_branches(fake_adapters):
    fake_adapters["url"].probability = 0.95
    fake_adapters["text"].probability = 0.05
    orch = Orchestrator(adapters=fake_adapters)
    orch.fusion_config = FusionConfig(disagreement_gap=0.5).validate()
    result = orch.analyze(CaseInput(url="https://example.com/", message_text="hi"))
    joined = " ".join(result.explanations)
    assert "disagree" in joined
    assert "URL" in joined
    assert "message text" in joined
