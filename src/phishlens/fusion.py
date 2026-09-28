"""Fusion of branch probabilities into one score.

Methods, selectable by name:
  learned      regularised logistic regression over six features
  mean         equal average of the available branch probabilities
  max          maximum of the available branch probabilities
  best_single  one named branch (falls back to the mean of the available
               branches when that branch has no usable score, and says so)

Feature vector for `learned`, in this exact order:
  [logit(p_url)*a_url, logit(p_text)*a_text, logit(p_image)*a_image,
   a_url, a_text, a_image]
where a_x is 1 when the branch is available and ok, else 0, and
logit(p) = ln(p / (1 - p)) with p clipped to [1e-6, 1 - 1e-6].

Calibration (optional) is applied to the logit z of the raw fused score:
  {"type": "temperature", "temperature": T}   score = sigmoid(z / T)
  {"type": "platt", "a": a, "b": b}           score = sigmoid(a * z + b)
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from . import config
from .schemas import BranchEvidence, FusionResult

BRANCH_ORDER = ("url", "text", "image")
FEATURE_ORDER = [
    "logit_p_url_x_a_url",
    "logit_p_text_x_a_text",
    "logit_p_image_x_a_image",
    "a_url",
    "a_text",
    "a_image",
]
METHODS = ("learned", "mean", "max", "best_single")
CLIP_EPS = 1e-6

DEFAULT_LOW = 0.30
DEFAULT_HIGH = 0.70
DEFAULT_GAP = 0.50

PROVISIONAL_NOTICE = (
    "Provisional settings in use. The fusion parameters and thresholds have not been "
    "fitted or validated on evaluation data. Treat the score and the outcome as indicative only."
)


class FusionConfigError(ValueError):
    pass


@dataclass
class FusionConfig:
    method: str = "mean"
    feature_order: List[str] = field(default_factory=lambda: list(FEATURE_ORDER))
    coefficients: Optional[List[float]] = None
    intercept: Optional[float] = None
    calibration: Optional[Dict[str, Any]] = None
    thresholds: Dict[str, float] = field(
        default_factory=lambda: {"low": DEFAULT_LOW, "high": DEFAULT_HIGH}
    )
    disagreement_gap: float = DEFAULT_GAP
    fitted: bool = False
    thresholds_provisional: bool = True
    best_single_branch: Optional[str] = None
    fitted_on: str = ""
    created: str = ""
    notes: str = ""

    @property
    def provisional(self) -> bool:
        """True when any setting in force has not been fitted or validated."""
        return (not self.fitted) or self.thresholds_provisional

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> Dict[str, Any]:
        data = self.to_dict()
        data["provisional"] = self.provisional
        return data

    def validate(self) -> "FusionConfig":
        if self.method not in METHODS:
            raise FusionConfigError(
                "Unknown fusion method %r. Choose one of: %s" % (self.method, ", ".join(METHODS))
            )
        if list(self.feature_order) != FEATURE_ORDER:
            raise FusionConfigError(
                "feature_order must be exactly %s" % json.dumps(FEATURE_ORDER)
            )
        low = float(self.thresholds.get("low", float("nan")))
        high = float(self.thresholds.get("high", float("nan")))
        if not (0.0 <= low < high <= 1.0):
            raise FusionConfigError(
                "thresholds must satisfy 0 <= low < high <= 1, got low=%r high=%r" % (low, high)
            )
        self.thresholds = {"low": low, "high": high}
        self.disagreement_gap = float(self.disagreement_gap)
        if not (0.0 <= self.disagreement_gap <= 1.0):
            raise FusionConfigError("disagreement_gap must be between 0 and 1")
        if self.method == "learned":
            if self.coefficients is None or len(self.coefficients) != len(FEATURE_ORDER):
                raise FusionConfigError(
                    "method 'learned' needs %d coefficients" % len(FEATURE_ORDER)
                )
            if self.intercept is None:
                raise FusionConfigError("method 'learned' needs an intercept")
            if not all(math.isfinite(float(c)) for c in self.coefficients):
                raise FusionConfigError("coefficients must be finite numbers")
        if self.method == "best_single" and self.best_single_branch not in BRANCH_ORDER:
            raise FusionConfigError(
                "method 'best_single' needs best_single_branch set to one of: %s"
                % ", ".join(BRANCH_ORDER)
            )
        if self.calibration:
            kind = self.calibration.get("type")
            if kind == "temperature":
                if float(self.calibration.get("temperature", 0.0)) <= 0.0:
                    raise FusionConfigError("temperature must be greater than 0")
            elif kind == "platt":
                float(self.calibration["a"])
                float(self.calibration["b"])
            else:
                raise FusionConfigError("calibration type must be 'temperature' or 'platt'")
        return self


# --------------------------------------------------------------------------
# Reading and writing fusion.json
# --------------------------------------------------------------------------
_KNOWN_KEYS = set(FusionConfig.__dataclass_fields__.keys())


def config_from_dict(raw: Mapping[str, Any]) -> FusionConfig:
    kwargs = {k: v for k, v in raw.items() if k in _KNOWN_KEYS}
    if "thresholds_provisional" not in kwargs:
        # Older or hand written files: thresholds count as provisional unless fitted.
        kwargs["thresholds_provisional"] = not bool(raw.get("fitted", False))
    return FusionConfig(**kwargs).validate()


def load_config(path: Optional[Path] = None) -> FusionConfig:
    """Read fusion.json. A missing file gives the unfitted provisional default."""
    path = Path(path) if path is not None else config.fusion_path()
    if not path.exists():
        cfg = FusionConfig(notes="fusion.json not found, built-in provisional defaults in use")
        return cfg.validate()
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return config_from_dict(raw)


def save_config(cfg: FusionConfig, path: Optional[Path] = None) -> Path:
    path = Path(path) if path is not None else config.fusion_path()
    cfg.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(cfg.to_dict(), handle, indent=2)
        handle.write("\n")
    return path


# --------------------------------------------------------------------------
# Maths
# --------------------------------------------------------------------------
def clipped_logit(p: float) -> float:
    p = min(1.0 - CLIP_EPS, max(CLIP_EPS, float(p)))
    return math.log(p / (1.0 - p))


def sigmoid(z: float) -> float:
    z = max(-60.0, min(60.0, float(z)))
    return 1.0 / (1.0 + math.exp(-z))


def _usable(value: Optional[float]) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(float(value))


def build_feature_vector(
    probabilities: Mapping[str, Optional[float]],
    availability: Optional[Mapping[str, Any]] = None,
) -> List[float]:
    """Six features in FEATURE_ORDER for one case.

    `probabilities` maps branch name to P(phishing) or None. `availability`
    maps branch name to a truthy flag. When omitted, a branch counts as
    available if it has a finite probability. A branch flagged available but
    without a finite probability counts as unavailable.
    """
    logits: List[float] = []
    flags: List[float] = []
    for branch in BRANCH_ORDER:
        p = probabilities.get(branch)
        ok = _usable(p)
        if availability is not None:
            ok = ok and bool(availability.get(branch, False))
        flags.append(1.0 if ok else 0.0)
        logits.append(clipped_logit(p) if ok else 0.0)
    return logits + flags


def build_feature_matrix(probabilities: Any, availability: Any = None) -> np.ndarray:
    """Batch version: probabilities (n, 3) in url, text, image order. NaN marks missing."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("probabilities must have shape (n, 3) in url, text, image order")
    finite = np.isfinite(p)
    if availability is None:
        a = finite
    else:
        a = np.asarray(availability).astype(bool)
        if a.shape != p.shape:
            raise ValueError("availability must have the same shape as probabilities")
        a = a & finite
    safe = np.clip(np.where(a, p, 0.5), CLIP_EPS, 1.0 - CLIP_EPS)
    logits = np.log(safe / (1.0 - safe)) * a
    return np.concatenate([logits, a.astype(np.float64)], axis=1)


def apply_calibration(score: float, calibration: Optional[Mapping[str, Any]]) -> float:
    if not calibration:
        return score
    z = clipped_logit(score)
    if calibration.get("type") == "temperature":
        return sigmoid(z / float(calibration["temperature"]))
    if calibration.get("type") == "platt":
        return sigmoid(float(calibration["a"]) * z + float(calibration["b"]))
    raise FusionConfigError("calibration type must be 'temperature' or 'platt'")


# --------------------------------------------------------------------------
# Fusing one case
# --------------------------------------------------------------------------
def fuse_probabilities(
    probabilities: Mapping[str, Optional[float]],
    cfg: FusionConfig,
    availability: Optional[Mapping[str, Any]] = None,
    method: Optional[str] = None,
) -> FusionResult:
    """Fuse one case. `method` overrides the method named in the config."""
    method = method or cfg.method
    if method not in METHODS:
        raise FusionConfigError("Unknown fusion method %r" % method)

    features = build_feature_vector(probabilities, availability)
    flags = features[3:]
    used = [b for b, flag in zip(BRANCH_ORDER, flags) if flag == 1.0]
    values = [float(probabilities[b]) for b in used]

    raw: Optional[float] = None
    note: Optional[str] = None
    if used:
        if method == "mean":
            raw = sum(values) / len(values)
        elif method == "max":
            raw = max(values)
        elif method == "best_single":
            branch = cfg.best_single_branch
            if branch not in BRANCH_ORDER:
                raise FusionConfigError("best_single needs best_single_branch in the config")
            if branch in used:
                raw = float(probabilities[branch])
                used = [branch]
            else:
                raw = sum(values) / len(values)
                note = (
                    "The configured single branch (%s) had no usable score, so the mean of "
                    "the available branches was used instead." % branch
                )
        elif method == "learned":
            if cfg.coefficients is None or cfg.intercept is None:
                raise FusionConfigError("method 'learned' needs coefficients and an intercept")
            z = float(cfg.intercept) + sum(
                float(c) * f for c, f in zip(cfg.coefficients, features)
            )
            raw = sigmoid(z)

    score = None if raw is None else min(1.0, max(0.0, apply_calibration(raw, cfg.calibration)))
    return FusionResult(
        method=method,
        score=score,
        raw_score=raw,
        fitted=cfg.fitted,
        provisional=cfg.provisional,
        feature_order=list(FEATURE_ORDER),
        features=features,
        branches_used=used,
        calibration=dict(cfg.calibration) if cfg.calibration else None,
        thresholds=dict(cfg.thresholds),
        disagreement_gap=cfg.disagreement_gap,
        fitted_on=cfg.fitted_on or None,
        note=note,
    )


def fuse(evidence: Iterable[BranchEvidence], cfg: FusionConfig) -> FusionResult:
    """Fuse the usable branches of one case."""
    probabilities: Dict[str, Optional[float]] = {b: None for b in BRANCH_ORDER}
    for item in evidence:
        if item.branch in probabilities and item.usable:
            probabilities[item.branch] = float(item.probability)
    return fuse_probabilities(probabilities, cfg)


def fuse_batch(
    probabilities: Any,
    availability: Any = None,
    cfg: Optional[FusionConfig] = None,
    method: Optional[str] = None,
) -> np.ndarray:
    """Fused scores for many cases. Rows with no available branch give NaN.

    probabilities: shape (n, 3) in url, text, image order, NaN where missing.
    """
    cfg = cfg or load_config()
    p = np.asarray(probabilities, dtype=np.float64)
    a = None if availability is None else np.asarray(availability)
    out = np.full(p.shape[0], np.nan, dtype=np.float64)
    for i in range(p.shape[0]):
        probs = {b: (float(p[i, j]) if np.isfinite(p[i, j]) else None)
                 for j, b in enumerate(BRANCH_ORDER)}
        avail = None if a is None else {b: bool(a[i, j]) for j, b in enumerate(BRANCH_ORDER)}
        result = fuse_probabilities(probs, cfg, availability=avail, method=method)
        if result.score is not None:
            out[i] = result.score
    return out


# --------------------------------------------------------------------------
# Fitting (called by the evaluation code)
# --------------------------------------------------------------------------
def fit(
    probabilities: Any,
    availability: Any,
    labels: Sequence[int],
    *,
    C: float = 1.0,
    class_weight: Optional[Any] = None,
    thresholds: Optional[Mapping[str, float]] = None,
    disagreement_gap: Optional[float] = None,
    calibration: Optional[Mapping[str, Any]] = None,
    fitted_on: str = "",
    notes: str = "",
    output_path: Optional[Path] = None,
    write: bool = True,
    max_iter: int = 1000,
) -> FusionConfig:
    """Fit the learned fusion and write fusion.json.

    probabilities  array (n, 3), columns in url, text, image order. Use NaN
                   (or any value with availability 0) for a missing branch.
    availability   array (n, 3) of 0/1 flags, or None to derive the flags
                   from which probabilities are finite.
    labels         array (n,), 1 phishing, 0 benign.
    thresholds     {"low": ..., "high": ...} chosen on validation data. When
                   omitted the provisional defaults are kept and the config is
                   marked thresholds_provisional, so the interface keeps
                   showing the provisional notice.
    output_path    where to write. Defaults to models/fusion.json.

    Returns the FusionConfig that was written.
    """
    from sklearn.linear_model import LogisticRegression

    X = build_feature_matrix(probabilities, availability)
    y = np.asarray(labels).astype(int).reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError("probabilities and labels have different lengths")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("labels must contain both classes, coded 0 and 1")
    if not np.any(X[:, 3:].sum(axis=1) > 0):
        raise ValueError("no row has an available branch")

    model = LogisticRegression(
        penalty="l2", C=C, class_weight=class_weight, max_iter=max_iter, solver="lbfgs"
    )
    model.fit(X, y)

    cfg = FusionConfig(
        method="learned",
        feature_order=list(FEATURE_ORDER),
        coefficients=[float(c) for c in model.coef_.reshape(-1)],
        intercept=float(model.intercept_.reshape(-1)[0]),
        calibration=dict(calibration) if calibration else None,
        thresholds=(
            {"low": float(thresholds["low"]), "high": float(thresholds["high"])}
            if thresholds
            else {"low": DEFAULT_LOW, "high": DEFAULT_HIGH}
        ),
        disagreement_gap=DEFAULT_GAP if disagreement_gap is None else float(disagreement_gap),
        fitted=True,
        thresholds_provisional=thresholds is None,
        fitted_on=fitted_on or "%d cases (%d phishing, %d benign)" % (
            len(y), int(y.sum()), int(len(y) - y.sum())
        ),
        created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        notes=notes or "Fitted with scikit-learn LogisticRegression (L2, C=%g)." % C,
    ).validate()
    if write:
        save_config(cfg, output_path)
    return cfg
