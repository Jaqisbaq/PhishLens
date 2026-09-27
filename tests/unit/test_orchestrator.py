"""Orchestrator: missing modality, failure isolation, end to end with fakes."""
from __future__ import annotations

from phishlens.fusion import FusionConfig
from phishlens.orchestrator import Orchestrator
from phishlens.schemas import STATUS_FAILED, STATUS_OK, STATUS_UNAVAILABLE, CaseInput


def make_orchestrator(fake_adapters, **cfg_kwargs):
    orch = Orchestrator(adapters=fake_adapters)
    orch.fusion_config = FusionConfig(**cfg_kwargs).validate()
    return orch


def test_all_three_branches_ok(fake_adapters):
    orch = make_orchestrator(fake_adapters)
    case = CaseInput(url="https://example.com/", message_text="hello", screenshot=object())
    result = orch.analyze(case)
    assert all(b.status == STATUS_OK for b in result.branches)
    assert result.fused_score is not None
    assert result.total_latency_ms is not None and result.total_latency_ms >= 0


def test_missing_modality_marks_branch_unavailable(fake_adapters):
    orch = make_orchestrator(fake_adapters)
    case = CaseInput(url="https://example.com/")  # text and image not provided
    result = orch.analyze(case)
    by_branch = {b.branch: b for b in result.branches}
    assert by_branch["url"].status == STATUS_OK
    assert by_branch["text"].status == STATUS_UNAVAILABLE
    assert by_branch["text"].input_provided is False
    assert by_branch["image"].status == STATUS_UNAVAILABLE
    assert fake_adapters["text"].predict_calls == 0
    assert fake_adapters["image"].predict_calls == 0


def test_failing_branch_is_isolated_others_still_run(fake_adapters):
    fake_adapters["text"].raise_on_predict = RuntimeError("boom")
    orch = make_orchestrator(fake_adapters)
    case = CaseInput(url="https://example.com/", message_text="hello", screenshot=object())
    result = orch.analyze(case)
    by_branch = {b.branch: b for b in result.branches}
    assert by_branch["text"].status == STATUS_FAILED
    assert "boom" in by_branch["text"].message
    assert by_branch["url"].status == STATUS_OK
    assert by_branch["image"].status == STATUS_OK
    # fusion should have used the two ok branches, not raised
    assert sorted(result.fusion.branches_used) == ["image", "url"]


def test_all_branches_fail_gives_unavailable_outcome(fake_adapters):
    for a in fake_adapters.values():
        a.raise_on_predict = RuntimeError("down")
    orch = make_orchestrator(fake_adapters)
    case = CaseInput(url="https://example.com/", message_text="hello", screenshot=object())
    result = orch.analyze(case)
    assert result.outcome == "Analysis unavailable"
    assert result.fused_score is None


def test_case_id_and_created_are_present(fake_adapters):
    orch = make_orchestrator(fake_adapters)
    result = orch.analyze(CaseInput(url="https://example.com/"))
    assert result.case_id
    assert result.created


def test_health_reports_branch_and_fusion_status(fake_adapters):
    orch = make_orchestrator(fake_adapters)
    health = orch.health()
    assert set(health["branches"]) == {"url", "text", "image"}
    assert health["fusion"]["method"] == orch.fusion_config.method
