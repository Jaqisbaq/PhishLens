"""Orchestrates the three branches, fuses their scores and applies policy.

Loads the three adapters once (lazily, on first use of each) and runs one
case through validation, the branches, fusion, policy and explanations.
A branch that raises is caught here and reported as "failed"; the other
branches still run. Nothing here ever opens the submitted URL.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from . import config, fusion
from .explanations import build_explanations
from .policy import decide
from .schemas import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    BranchEvidence,
    CaseInput,
    CaseResult,
)


def _unavailable(branch: str, message: str) -> BranchEvidence:
    return BranchEvidence(
        branch=branch,
        status=STATUS_UNAVAILABLE,
        input_provided=False,
        message=message,
    )


def _failed(adapter, branch: str, exc: Exception) -> BranchEvidence:
    return BranchEvidence(
        branch=branch,
        status=STATUS_FAILED,
        input_provided=True,
        model_id=getattr(adapter, "model_id", None),
        revision=getattr(adapter, "revision", None),
        message="%s: %s" % (type(exc).__name__, exc),
    )


class Orchestrator:
    """Owns the three adapters and the fusion configuration."""

    def __init__(self, adapters: Optional[Dict[str, Any]] = None) -> None:
        if adapters is None:
            from .adapters import build_default_adapters

            adapters = build_default_adapters()
        self.adapters = adapters
        self.fusion_config = fusion.load_config()

    # ---- setup -----------------------------------------------------------
    def preload(self) -> None:
        """Load all three models now, instead of on first request."""
        for adapter in self.adapters.values():
            adapter.load()

    def reload_fusion_config(self) -> fusion.FusionConfig:
        self.fusion_config = fusion.load_config()
        return self.fusion_config

    # ---- per-case run ------------------------------------------------------
    def _run_branch(self, branch: str, case: CaseInput) -> BranchEvidence:
        adapter = self.adapters.get(branch)
        value = {"url": case.url, "text": case.message_text, "image": case.screenshot}[branch]
        if value is None:
            return _unavailable(branch, "No %s was provided." % branch)
        if adapter is None:
            return _unavailable(branch, "No adapter is configured for this branch.")
        try:
            return adapter.predict(value)
        except Exception as exc:  # noqa: BLE001 - isolate branch failures
            return _failed(adapter, branch, exc)

    def analyze(self, case: CaseInput) -> CaseResult:
        t0 = time.perf_counter()
        branches = [self._run_branch(b, case) for b in config.BRANCHES]

        probabilities = {b.branch: (b.probability if b.usable else None) for b in branches}
        fused = fusion.fuse_probabilities(probabilities, self.fusion_config)
        decision = decide(
            fused.score,
            probabilities,
            low=self.fusion_config.thresholds["low"],
            high=self.fusion_config.thresholds["high"],
            disagreement_gap=self.fusion_config.disagreement_gap,
        )
        explanations = build_explanations(branches, fused, decision)

        provisional_notice = fusion.PROVISIONAL_NOTICE if fused.provisional else None
        total_latency = round((time.perf_counter() - t0) * 1000.0, 2)

        return CaseResult(
            case_id=uuid.uuid4().hex,
            created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            outcome=decision.outcome,
            fused_score=fused.score,
            decision=decision,
            fusion=fused,
            branches=branches,
            inputs_provided=case.provided(),
            input_summary=_input_summary(case),
            explanations=explanations,
            provisional_settings=fused.provisional,
            provisional_notice=provisional_notice,
            total_latency_ms=total_latency,
        )

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "branches": {name: a.describe() for name, a in self.adapters.items()},
            "fusion": self.fusion_config.summary(),
        }


def _input_summary(case: CaseInput) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    if case.url is not None:
        summary["url_length"] = len(case.url)
    if case.message_text is not None:
        summary["text_length"] = len(case.message_text)
    if case.screenshot_info:
        summary["screenshot"] = dict(case.screenshot_info)
    return summary
