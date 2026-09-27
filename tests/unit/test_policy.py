"""Policy: threshold boundaries, disagreement gate, unavailable outcome."""
from __future__ import annotations

from phishlens.policy import (
    RULE_BETWEEN,
    RULE_DISAGREEMENT,
    RULE_HIGH,
    RULE_LOW,
    RULE_UNAVAILABLE,
    decide,
)
from phishlens.schemas import OUTCOME_HIGH, OUTCOME_LIMITED, OUTCOME_REVIEW, OUTCOME_UNAVAILABLE

LOW, HIGH, GAP = 0.30, 0.70, 0.50


def test_no_usable_branch_gives_unavailable():
    result = decide(None, {"url": None, "text": None, "image": None}, LOW, HIGH, GAP)
    assert result.outcome == OUTCOME_UNAVAILABLE
    assert result.rule == RULE_UNAVAILABLE


def test_score_above_high_is_high_concern():
    result = decide(0.85, {"url": 0.85, "text": None, "image": None}, LOW, HIGH, GAP)
    assert result.outcome == OUTCOME_HIGH
    assert result.rule == RULE_HIGH


def test_score_exactly_high_boundary_is_high_concern():
    result = decide(HIGH, {"url": HIGH, "text": None, "image": None}, LOW, HIGH, GAP)
    assert result.outcome == OUTCOME_HIGH


def test_score_exactly_low_boundary_is_limited():
    result = decide(LOW, {"url": LOW, "text": None, "image": None}, LOW, HIGH, GAP)
    assert result.outcome == OUTCOME_LIMITED
    assert result.rule == RULE_LOW


def test_score_below_low_is_limited():
    result = decide(0.1, {"url": 0.1, "text": None, "image": None}, LOW, HIGH, GAP)
    assert result.outcome == OUTCOME_LIMITED


def test_score_between_thresholds_is_review():
    result = decide(0.5, {"url": 0.5, "text": None, "image": None}, LOW, HIGH, GAP)
    assert result.outcome == OUTCOME_REVIEW
    assert result.rule == RULE_BETWEEN


def test_disagreement_gate_overrides_high_score():
    # url=0.95, text=0.1: gap 0.85 > 0.5, forces review even though fused score is high.
    result = decide(0.9, {"url": 0.95, "text": 0.1, "image": None}, LOW, HIGH, GAP)
    assert result.outcome == OUTCOME_REVIEW
    assert result.rule == RULE_DISAGREEMENT
    assert result.disagreement is True
    assert result.branch_gap == 0.85


def test_disagreement_gate_needs_two_usable_branches():
    result = decide(0.9, {"url": 0.95, "text": None, "image": None}, LOW, HIGH, GAP)
    assert result.rule == RULE_HIGH


def test_gap_exactly_at_limit_does_not_trigger_gate():
    result = decide(0.9, {"url": 0.8, "text": 0.3, "image": None}, LOW, HIGH, 0.5)
    assert result.rule == RULE_HIGH


def test_gap_just_above_limit_triggers_gate():
    result = decide(0.9, {"url": 0.801, "text": 0.3, "image": None}, LOW, HIGH, 0.5)
    assert result.rule == RULE_DISAGREEMENT


def test_outcome_text_never_says_safe():
    for outcome in (OUTCOME_HIGH, OUTCOME_LIMITED, OUTCOME_REVIEW, OUTCOME_UNAVAILABLE):
        assert "safe" not in outcome.lower()
