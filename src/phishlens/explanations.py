"""Template-based explanations.

Every sentence is built from a value that is present in the evidence. There
is no free text generation and no claim that the evidence does not support.
"""
from __future__ import annotations

from typing import List, Sequence

from .fusion import PROVISIONAL_NOTICE
from .policy import (
    RULE_BETWEEN,
    RULE_DISAGREEMENT,
    RULE_HIGH,
    RULE_LOW,
    RULE_UNAVAILABLE,
)
from .schemas import (
    STATUS_FAILED,
    STATUS_OK,
    BranchEvidence,
    FusionResult,
    PolicyDecision,
)

BRANCH_LABELS = {"url": "URL", "text": "message text", "image": "screenshot"}

METHOD_LABELS = {
    "mean": "the equal average of the available branch scores",
    "max": "the highest of the available branch scores",
    "best_single": "a single configured branch",
    "learned": "a fitted logistic regression over the branch scores",
}


def _label(branch: str) -> str:
    return BRANCH_LABELS.get(branch, branch)


def _fmt(value: float) -> str:
    return "%.2f" % value


def _url_feature_sentences(evidence: BranchEvidence) -> List[str]:
    d = evidence.details
    out: List[str] = []
    present: List[str] = []
    if d.get("host_is_ip_address"):
        present.append("the host is a numeric IP address rather than a name")
    if d.get("has_at_symbol"):
        present.append("it contains an '@' character")
    if d.get("uses_https") is False:
        present.append("it does not use https")
    if d.get("host_is_punycode"):
        present.append("the host uses punycode (xn--) encoding")
    hyphens = d.get("hyphen_count")
    if isinstance(hyphens, int) and hyphens >= 3:
        present.append("it contains %d hyphens" % hyphens)
    dots = d.get("dot_count")
    if isinstance(dots, int) and dots >= 5:
        present.append("it contains %d dots" % dots)
    length = d.get("length")
    if isinstance(length, int) and length >= 100:
        present.append("it is %d characters long" % length)
    if present:
        out.append(
            "Structural features of the URL string: " + "; ".join(present) + ". "
            "These are descriptive only and were computed from the text of the URL."
        )
    return out


def build_explanations(
    branches: Sequence[BranchEvidence],
    fusion: FusionResult,
    decision: PolicyDecision,
) -> List[str]:
    sentences: List[str] = []
    usable = [b for b in branches if b.usable]

    # 1. Outcome and the rule that produced it.
    if decision.rule == RULE_UNAVAILABLE:
        sentences.append(
            "No branch produced a usable score, so no assessment could be made."
        )
    elif decision.rule == RULE_DISAGREEMENT:
        sentences.append(
            "The evidence sources disagree: the %s branch scored %s and the %s branch scored %s, "
            "a gap of %s, which is larger than the disagreement limit of %s. "
            "The case is marked for review regardless of the fused score of %s."
            % (
                _label(decision.highest_branch),
                _fmt(next(b.probability for b in usable if b.branch == decision.highest_branch)),
                _label(decision.lowest_branch),
                _fmt(next(b.probability for b in usable if b.branch == decision.lowest_branch)),
                _fmt(decision.branch_gap),
                _fmt(fusion.disagreement_gap),
                _fmt(fusion.score),
            )
        )
    elif decision.rule == RULE_HIGH:
        sentences.append(
            "The fused score is %s, which is at or above the high threshold of %s."
            % (_fmt(fusion.score), _fmt(fusion.thresholds["high"]))
        )
    elif decision.rule == RULE_LOW:
        sentences.append(
            "The fused score is %s, which is at or below the low threshold of %s. "
            "The models found few indicators in what was submitted."
            % (_fmt(fusion.score), _fmt(fusion.thresholds["low"]))
        )
    elif decision.rule == RULE_BETWEEN:
        sentences.append(
            "The fused score is %s, which lies between the low threshold of %s and the "
            "high threshold of %s." % (
                _fmt(fusion.score), _fmt(fusion.thresholds["low"]), _fmt(fusion.thresholds["high"])
            )
        )

    # 2. How the score was combined.
    if fusion.score is not None:
        sentences.append(
            "The score was combined using %s, from %d of 3 branches (%s)."
            % (
                METHOD_LABELS.get(fusion.method, fusion.method),
                len(fusion.branches_used),
                ", ".join(_label(b) for b in fusion.branches_used),
            )
        )
    if fusion.note:
        sentences.append(fusion.note)

    # 3. Which branch contributed the highest score.
    if len(usable) >= 2:
        top = max(usable, key=lambda b: b.probability)
        sentences.append(
            "The highest branch score came from the %s branch (%s)."
            % (_label(top.branch), _fmt(top.probability))
        )
    elif len(usable) == 1:
        only = usable[0]
        sentences.append(
            "Only the %s branch produced a score (%s), so the result rests on a single source."
            % (_label(only.branch), _fmt(only.probability))
        )

    # 4. Missing, unavailable and failed branches.
    for b in branches:
        if b.status == STATUS_OK:
            continue
        if not b.input_provided:
            sentences.append(
                "No %s was provided, so the %s branch did not contribute."
                % (_label(b.branch), _label(b.branch))
            )
        elif b.status == STATUS_FAILED:
            sentences.append(
                "The %s branch failed during analysis and did not contribute." % _label(b.branch)
            )
        else:
            sentences.append(
                "The %s branch was unavailable and did not contribute." % _label(b.branch)
            )

    # 5. Truncation.
    for b in usable:
        if b.truncated:
            tokens = b.details.get("token_count")
            limit = b.details.get("max_tokens")
            if tokens is not None and limit is not None:
                sentences.append(
                    "The %s was longer than the model limit (%s tokens, limit %s), so only "
                    "the first part was analysed." % (_label(b.branch), tokens, limit)
                )
            else:
                sentences.append(
                    "The %s was longer than the model limit, so only the first part was analysed."
                    % _label(b.branch)
                )

    # 6. Branch specific details.
    for b in usable:
        if b.branch == "url":
            sentences.extend(_url_feature_sentences(b))
        elif b.branch == "image":
            mode = b.details.get("mode")
            if mode == "zero_shot" and b.details.get("top_prompt"):
                sentences.append(
                    "The screenshot was scored by zero-shot prompt matching. The closest prompt "
                    "was \"%s\" (%s)." % (
                        b.details["top_prompt"], _fmt(float(b.details["top_prompt_score"]))
                    )
                )
            elif mode == "fitted_head":
                sentences.append(
                    "The screenshot was scored by a fitted logistic head over the image embedding."
                )

    # 7. Provisional settings.
    if fusion.provisional:
        sentences.append(PROVISIONAL_NOTICE)

    return sentences
