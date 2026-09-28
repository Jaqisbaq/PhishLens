"""Decision policy: thresholds, three outcomes, disagreement and abstention gates.

Order of the rules:
  1. No usable branch score or no fused score    -> "Analysis unavailable"
  2. Disagreement gate: two or more usable branches and the gap between the
     highest and the lowest branch probability exceeds `disagreement_gap`
                                                   -> "Review needed"
  3. fused score >= high                           -> "High concern"
  4. fused score <= low                            -> "Limited indicators"
  5. otherwise                                     -> "Review needed"

No outcome ever states that an item is legitimate.
"""
from __future__ import annotations

from typing import Mapping, Optional

from .schemas import (
    OUTCOME_HIGH,
    OUTCOME_LIMITED,
    OUTCOME_REVIEW,
    OUTCOME_UNAVAILABLE,
    PolicyDecision,
)

RULE_UNAVAILABLE = "no_usable_score"
RULE_DISAGREEMENT = "disagreement_gate"
RULE_HIGH = "score_at_or_above_high"
RULE_LOW = "score_at_or_below_low"
RULE_BETWEEN = "score_between_thresholds"

BRANCH_ORDER = ("url", "text", "image")


def decide(
    fused_score: Optional[float],
    branch_probabilities: Mapping[str, Optional[float]],
    low: float,
    high: float,
    disagreement_gap: float,
) -> PolicyDecision:
    """Turn a fused score and the usable branch probabilities into an outcome."""
    usable = {
        b: float(branch_probabilities[b])
        for b in BRANCH_ORDER
        if branch_probabilities.get(b) is not None
    }
    if not usable or fused_score is None:
        return PolicyDecision(outcome=OUTCOME_UNAVAILABLE, rule=RULE_UNAVAILABLE)

    highest = max(usable, key=lambda b: usable[b])
    lowest = min(usable, key=lambda b: usable[b])
    gap = usable[highest] - usable[lowest] if len(usable) >= 2 else None

    common = dict(
        branch_gap=None if gap is None else round(gap, 6),
        highest_branch=highest,
        lowest_branch=lowest if len(usable) >= 2 else None,
    )

    if gap is not None and gap > disagreement_gap:
        return PolicyDecision(
            outcome=OUTCOME_REVIEW, rule=RULE_DISAGREEMENT, disagreement=True, **common
        )
    if fused_score >= high:
        return PolicyDecision(outcome=OUTCOME_HIGH, rule=RULE_HIGH, **common)
    if fused_score <= low:
        return PolicyDecision(outcome=OUTCOME_LIMITED, rule=RULE_LOW, **common)
    return PolicyDecision(outcome=OUTCOME_REVIEW, rule=RULE_BETWEEN, **common)
