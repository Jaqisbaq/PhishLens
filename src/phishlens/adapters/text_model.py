"""Text branch: BERT classifier over the message text."""
from __future__ import annotations

from typing import List, Sequence

from .. import config
from ..schemas import BranchEvidence
from .base import BaseAdapter, load_with_retry


class TextModelAdapter(BaseAdapter):
    name = "text"
    model_id = config.TEXT_MODEL_ID
    revision = config.TEXT_MODEL_REVISION
    max_tokens = config.TEXT_MAX_TOKENS
    phishing_index = config.TEXT_PHISHING_INDEX

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

    def predict_batch(self, values: Sequence[str], batch_size: int = 8) -> List[float]:
        """P(phishing) for each message. Inputs beyond 512 tokens are truncated."""
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
        return BranchEvidence(
            branch=self.name,
            probability=min(1.0, max(0.0, probability)),
            truncated=token_count > self.max_tokens,
            details={
                "token_count": token_count,
                "max_tokens": self.max_tokens,
                "character_count": len(value),
                "label_mapping": "0 benign, 1 phishing (from the model config)",
            },
        )
