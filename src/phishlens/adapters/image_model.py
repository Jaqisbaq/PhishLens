"""Image branch: CLIP image embeddings.

Two modes:
  fitted_head  models/image_head.json exists. A logistic regression over the
               512-dim L2-normalised CLIP image embedding gives P(phishing).
  zero_shot    no head file. The score is the softmax mass that falls on a
               small set of suspicious prompts versus ordinary prompts. The
               prompt embeddings are encoded once and cached.

image_head.json format:
  {
    "coefficients": [512 numbers],
    "intercept": number,
    "scaler": {"mean": [512 numbers], "scale": [512 numbers]}   (optional)
    "fitted_on": "free text", "created": "date"                 (optional)
  }
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .. import config
from ..schemas import BranchEvidence
from .base import BaseAdapter, load_with_retry

MODE_ZERO_SHOT = "zero_shot"
MODE_FITTED_HEAD = "fitted_head"


def load_image_head(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Read and check image_head.json. Returns None when the file is absent."""
    path = Path(path) if path is not None else config.image_head_path()
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    coef = np.asarray(raw["coefficients"], dtype=np.float64).reshape(-1)
    if coef.shape[0] != config.IMAGE_EMBEDDING_DIM:
        raise ValueError(
            "image_head.json has %d coefficients, expected %d"
            % (coef.shape[0], config.IMAGE_EMBEDDING_DIM)
        )
    head: Dict[str, Any] = {
        "coefficients": coef,
        "intercept": float(np.asarray(raw["intercept"]).reshape(-1)[0]),
        "scaler_mean": None,
        "scaler_scale": None,
        "fitted_on": raw.get("fitted_on"),
        "created": raw.get("created"),
    }
    scaler = raw.get("scaler")
    if scaler:
        mean = np.asarray(scaler["mean"], dtype=np.float64).reshape(-1)
        scale = np.asarray(scaler["scale"], dtype=np.float64).reshape(-1)
        if mean.shape[0] != coef.shape[0] or scale.shape[0] != coef.shape[0]:
            raise ValueError("image_head.json scaler length does not match the coefficients")
        head["scaler_mean"] = mean
        head["scaler_scale"] = np.where(scale == 0.0, 1.0, scale)
    return head


def head_probability(embedding: np.ndarray, head: Dict[str, Any]) -> float:
    """P(phishing) from one L2-normalised embedding and a loaded head."""
    x = np.asarray(embedding, dtype=np.float64).reshape(-1)
    if head.get("scaler_mean") is not None:
        x = (x - head["scaler_mean"]) / head["scaler_scale"]
    z = float(np.dot(head["coefficients"], x) + head["intercept"])
    z = max(-60.0, min(60.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def zero_shot_scores(
    embedding: np.ndarray, prompt_embeddings: np.ndarray, logit_scale: float
) -> np.ndarray:
    """Softmax over the prompts for one L2-normalised image embedding."""
    logits = logit_scale * (prompt_embeddings @ np.asarray(embedding, dtype=np.float64))
    logits = logits - logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


class ImageModelAdapter(BaseAdapter):
    name = "image"
    model_id = config.IMAGE_MODEL_ID
    revision = config.IMAGE_MODEL_REVISION

    def __init__(self, head_path: Optional[Path] = None) -> None:
        super().__init__()
        self._head_path = head_path
        self.head: Optional[Dict[str, Any]] = None
        self.mode = MODE_ZERO_SHOT
        self.prompts: List[str] = list(config.SUSPICIOUS_PROMPTS) + list(config.ORDINARY_PROMPTS)
        self.suspicious_count = len(config.SUSPICIOUS_PROMPTS)

    def _load(self) -> None:
        config.configure_environment()
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self._processor = load_with_retry(
            lambda: CLIPProcessor.from_pretrained(self.model_id, revision=self.revision)
        )
        self._model = load_with_retry(
            lambda: CLIPModel.from_pretrained(self.model_id, revision=self.revision)
        )
        self._model.eval()
        self._logit_scale = float(self._model.logit_scale.exp().item())

        # Encode the prompt set once and cache it.
        enc = self._processor(text=self.prompts, return_tensors="pt", padding=True)
        with torch.no_grad():
            text = self._model.get_text_features(**enc)
        text = text / text.norm(dim=-1, keepdim=True)
        self._prompt_embeddings = text.numpy().astype(np.float64)

        self.reload_head()

    def reload_head(self) -> str:
        """Re-read image_head.json. Returns the mode now in force."""
        self.head = load_image_head(self._head_path)
        self.mode = MODE_FITTED_HEAD if self.head is not None else MODE_ZERO_SHOT
        return self.mode

    # ---- embeddings ------------------------------------------------------
    def _embed(self, images: Sequence[Any]) -> np.ndarray:
        torch = self._torch
        rgb = [im if getattr(im, "mode", "RGB") == "RGB" else im.convert("RGB") for im in images]
        enc = self._processor(images=rgb, return_tensors="pt")
        with torch.no_grad():
            feats = self._model.get_image_features(**enc)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.numpy().astype(np.float32)

    def embed_image(self, pil_image: Any) -> np.ndarray:
        """L2-normalised 512-dim CLIP image embedding, shape (512,), float32."""
        self.load()
        with self._infer_lock:
            return self._embed([pil_image])[0]

    def embed_images(self, pil_images: Sequence[Any], batch_size: int = 16) -> np.ndarray:
        """Batch version of embed_image. Returns shape (n, 512), float32."""
        self.load()
        images = list(pil_images)
        if not images:
            return np.zeros((0, config.IMAGE_EMBEDDING_DIM), dtype=np.float32)
        chunks = []
        with self._infer_lock:
            for start in range(0, len(images), batch_size):
                chunks.append(self._embed(images[start:start + batch_size]))
        return np.concatenate(chunks, axis=0)

    # ---- scoring ---------------------------------------------------------
    def score_embedding(self, embedding: np.ndarray, mode: Optional[str] = None) -> Dict[str, Any]:
        """Score one embedding. `mode` forces zero_shot or fitted_head."""
        self.load()
        mode = mode or self.mode
        if mode == MODE_FITTED_HEAD:
            if self.head is None:
                raise RuntimeError("fitted_head mode requested but no image_head.json is loaded")
            return {
                "mode": MODE_FITTED_HEAD,
                "probability": head_probability(embedding, self.head),
                "head_fitted_on": self.head.get("fitted_on"),
            }
        scores = zero_shot_scores(embedding, self._prompt_embeddings, self._logit_scale)
        top = int(np.argmax(scores))
        return {
            "mode": MODE_ZERO_SHOT,
            "probability": float(scores[: self.suspicious_count].sum()),
            "top_prompt": self.prompts[top],
            "top_prompt_score": round(float(scores[top]), 4),
            "top_prompt_group": "suspicious" if top < self.suspicious_count else "ordinary",
            "prompt_count": len(self.prompts),
        }

    def predict_batch(self, pil_images: Sequence[Any], mode: Optional[str] = None) -> List[float]:
        """P(phishing) for each image using the mode in force (or the one given)."""
        embeddings = self.embed_images(pil_images)
        return [float(self.score_embedding(e, mode)["probability"]) for e in embeddings]

    def _predict(self, value: Any) -> BranchEvidence:
        embedding = self._embed([value])[0]
        scored = self.score_embedding(embedding)
        probability = float(scored.pop("probability"))
        details = dict(scored)
        details["embedding_dim"] = int(embedding.shape[0])
        details["image_size"] = list(getattr(value, "size", ()))
        return BranchEvidence(
            branch=self.name,
            probability=min(1.0, max(0.0, probability)),
            truncated=False,
            details=details,
        )


# --------------------------------------------------------------------------
# Module level helpers for the evaluation code
# --------------------------------------------------------------------------
_default_adapter: Optional[ImageModelAdapter] = None


def get_default_adapter() -> ImageModelAdapter:
    """Shared ImageModelAdapter, created and loaded on first use."""
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = ImageModelAdapter()
    _default_adapter.load()
    return _default_adapter


def embed_image(pil_image: Any) -> np.ndarray:
    """L2-normalised 512-dim CLIP image embedding for one PIL image."""
    return get_default_adapter().embed_image(pil_image)


def embed_images(pil_images: Sequence[Any], batch_size: int = 16) -> np.ndarray:
    """L2-normalised CLIP image embeddings, shape (n, 512)."""
    return get_default_adapter().embed_images(pil_images, batch_size=batch_size)
