"""Data structures passed between the layers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

STATUS_OK = "ok"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"

OUTCOME_HIGH = "High concern"
OUTCOME_REVIEW = "Review needed"
OUTCOME_LIMITED = "Limited indicators"
OUTCOME_UNAVAILABLE = "Analysis unavailable"

CAVEAT = (
    "This is a triage aid, not a verdict. 'Limited indicators' means the models "
    "found few signs of phishing in what was submitted. It does not mean the item "
    "is legitimate. Always verify through a trusted channel before acting."
)


@dataclass
class CaseInput:
    """A validated case. All three parts are optional, at least one is present.

    `url` is the exact string the user submitted (only surrounding whitespace
    removed). `screenshot` is an RGB PIL image already opened and checked.
    """

    url: Optional[str] = None
    message_text: Optional[str] = None
    screenshot: Any = None  # PIL.Image.Image or None
    screenshot_info: Dict[str, Any] = field(default_factory=dict)

    def provided(self) -> Dict[str, bool]:
        return {
            "url": self.url is not None,
            "text": self.message_text is not None,
            "image": self.screenshot is not None,
        }


class _Model(BaseModel):
    # "model_id" is a plain field name here, not a pydantic internal.
    model_config = ConfigDict(protected_namespaces=())


class BranchEvidence(_Model):
    branch: str
    status: str = STATUS_UNAVAILABLE
    input_provided: bool = False
    probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    model_id: Optional[str] = None
    revision: Optional[str] = None
    latency_ms: Optional[float] = None
    truncated: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = None

    @property
    def usable(self) -> bool:
        return self.status == STATUS_OK and self.probability is not None


class FusionResult(_Model):
    method: str
    score: Optional[float] = None
    raw_score: Optional[float] = None
    fitted: bool = False
    provisional: bool = True
    feature_order: List[str] = Field(default_factory=list)
    features: List[float] = Field(default_factory=list)
    branches_used: List[str] = Field(default_factory=list)
    calibration: Optional[Dict[str, Any]] = None
    thresholds: Dict[str, float] = Field(default_factory=dict)
    disagreement_gap: Optional[float] = None
    fitted_on: Optional[str] = None
    note: Optional[str] = None


class PolicyDecision(_Model):
    outcome: str
    rule: str
    disagreement: bool = False
    branch_gap: Optional[float] = None
    highest_branch: Optional[str] = None
    lowest_branch: Optional[str] = None


class CaseResult(_Model):
    case_id: str
    created: str
    outcome: str
    fused_score: Optional[float] = None
    decision: PolicyDecision
    fusion: FusionResult
    branches: List[BranchEvidence]
    inputs_provided: Dict[str, bool]
    input_summary: Dict[str, Any] = Field(default_factory=dict)
    explanations: List[str] = Field(default_factory=list)
    provisional_settings: bool = True
    provisional_notice: Optional[str] = None
    caveat: str = CAVEAT
    total_latency_ms: Optional[float] = None
