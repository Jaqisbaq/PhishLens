"""Model adapters, one per evidence branch.

The adapter modules import torch and transformers lazily inside load(), so
importing this package is cheap and the unit tests never load a model.
"""
from .base import BaseAdapter, load_with_retry

__all__ = ["BaseAdapter", "load_with_retry", "build_default_adapters"]


def build_default_adapters():
    """Return the three real adapters keyed by branch name (not yet loaded)."""
    from .image_model import ImageModelAdapter
    from .text_model import TextModelAdapter
    from .url_model import UrlModelAdapter

    return {
        "url": UrlModelAdapter(),
        "text": TextModelAdapter(),
        "image": ImageModelAdapter(),
    }
