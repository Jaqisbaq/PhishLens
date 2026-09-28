"""URL branch: a small BERT classifier over the URL string.

The URL is only ever tokenised as text. It is never fetched, resolved or
opened. The full URL including its scheme is passed to the model, because
removing the scheme changes the scores of this model substantially.
"""
from __future__ import annotations

import ipaddress
from typing import Any, Dict, List, Sequence
from urllib.parse import urlsplit

from .. import config
from ..schemas import BranchEvidence
from .base import BaseAdapter, load_with_retry


def url_features(url: str) -> Dict[str, Any]:
    """Structural features computed from the string only (no lookup of any kind)."""
    features: Dict[str, Any] = {
        "length": len(url),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "digit_count": sum(ch.isdigit() for ch in url),
        "has_at_symbol": "@" in url,
        "uses_https": url.lower().startswith("https://"),
        "host_is_ip_address": False,
        "host_label_count": 0,
        "has_query": False,
        "has_explicit_port": False,
        "host_is_punycode": False,
    }
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        features["has_query"] = bool(parts.query)
        try:
            features["has_explicit_port"] = parts.port is not None
        except ValueError:
            features["has_explicit_port"] = True  # present but not a valid number
    except ValueError:
        return features
    if host:
        try:
            ipaddress.ip_address(host)  # pure string parsing, no network
            features["host_is_ip_address"] = True
        except ValueError:
            features["host_is_ip_address"] = False
        features["host_label_count"] = (
            0 if features["host_is_ip_address"] else len([p for p in host.split(".") if p])
        )
        features["host_is_punycode"] = any(
            label.startswith("xn--") for label in host.lower().split(".")
        )
    return features


class UrlModelAdapter(BaseAdapter):
    name = "url"
    model_id = config.URL_MODEL_ID
    revision = config.URL_MODEL_REVISION
    max_tokens = config.URL_MAX_TOKENS
    phishing_index = config.URL_PHISHING_INDEX

    def _load(self) -> None:
        config.configure_environment()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tokenizer = load_with_retry(
            lambda: AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        )
        self._model = load_with_retry(
            lambda: AutoModelForSequenceClassification.from_pretrained(
                self.model_id, revision=self.revision
            )
        )
        self._model.eval()

    # ---- helpers ---------------------------------------------------------
    def count_tokens(self, value: str) -> int:
        """Token count including special tokens, without truncation."""
        self.load()
        return len(self._tokenizer(value, truncation=False, add_special_tokens=True)["input_ids"])

    def _probabilities(self, values: Sequence[str]) -> List[float]:
        torch = self._torch
        enc = self._tokenizer(
            list(values),
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
            padding=True,
        )
        with torch.no_grad():
            logits = self._model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, self.phishing_index]
        return [float(p) for p in probs.tolist()]

    # ---- public API ------------------------------------------------------
    def predict_batch(self, values: Sequence[str], batch_size: int = 64) -> List[float]:
        """P(phishing) for each full URL string. Pass URLs with their scheme."""
        self.load()
        out: List[float] = []
        values = list(values)
        with self._infer_lock:
            for start in range(0, len(values), batch_size):
                out.extend(self._probabilities(values[start:start + batch_size]))
        return out

    def _predict(self, value: str) -> BranchEvidence:
        token_count = self.count_tokens(value)
        probability = self._probabilities([value])[0]
        details = dict(url_features(value))
        details["token_count"] = token_count
        details["max_tokens"] = self.max_tokens
        details["label_mapping"] = "0 good, 1 phishing (from the model card)"
        return BranchEvidence(
            branch=self.name,
            probability=min(1.0, max(0.0, probability)),
            truncated=token_count > self.max_tokens,
            details=details,
        )
