"""Input validation and normalisation.

The URL is checked as a string only. Nothing here performs a DNS lookup or
opens a connection. The URL is never rewritten: the adapter receives exactly
what the user submitted, apart from surrounding whitespace.
"""
from __future__ import annotations

import io
from typing import List, Optional
from urllib.parse import urlsplit

from . import config
from .schemas import CaseInput


class ValidationError(Exception):
    """Raised with a list of {"field": ..., "message": ...} entries."""

    def __init__(self, errors: List[dict]):
        self.errors = errors
        super().__init__("; ".join(e["field"] + ": " + e["message"] for e in errors))


def _blank(value: Optional[str]) -> bool:
    return value is None or value.strip() == ""


def validate_url(url: str) -> str:
    """Return the URL with surrounding whitespace removed, or raise ValueError."""
    candidate = url.strip()
    if len(candidate) > config.MAX_URL_LENGTH:
        raise ValueError(
            "The URL is %d characters long. The limit is %d."
            % (len(candidate), config.MAX_URL_LENGTH)
        )
    if any(ch.isspace() for ch in candidate) or any(ord(ch) < 32 for ch in candidate):
        raise ValueError("The URL must not contain spaces or control characters.")
    try:
        parts = urlsplit(candidate)
        host = parts.hostname
    except ValueError:
        raise ValueError("The URL could not be parsed.") from None
    if parts.scheme.lower() not in config.ALLOWED_URL_SCHEMES:
        raise ValueError(
            "The URL must start with http:// or https://. "
            "Enter the full address including the scheme."
        )
    if not host:
        raise ValueError("The URL has no host name.")
    return candidate


def validate_text(text: str) -> str:
    if len(text) > config.MAX_TEXT_LENGTH:
        raise ValueError(
            "The message is %d characters long. The limit is %d."
            % (len(text), config.MAX_TEXT_LENGTH)
        )
    return text.strip()


def validate_image(data: bytes):
    """Open the image with Pillow and return (rgb_image, info), or raise ValueError."""
    from PIL import Image, UnidentifiedImageError

    if len(data) == 0:
        raise ValueError("The screenshot file is empty.")
    if len(data) > config.MAX_IMAGE_BYTES:
        raise ValueError(
            "The screenshot is %.1f MB. The limit is %.0f MB."
            % (len(data) / (1024 * 1024), config.MAX_IMAGE_BYTES / (1024 * 1024))
        )
    try:
        # Image.open reads the header only, so the size check below runs
        # before any pixel data is decoded.
        probe = Image.open(io.BytesIO(data))
        fmt = probe.format
        width, height = probe.size
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise ValueError("The screenshot could not be opened as an image.") from None
    if fmt not in config.ALLOWED_IMAGE_FORMATS:
        raise ValueError("The screenshot must be a PNG or JPEG image, not %s." % fmt)
    if width < 1 or height < 1:
        raise ValueError("The screenshot has no pixels.")
    if (
        width * height > config.MAX_IMAGE_PIXELS
        or width > config.MAX_IMAGE_SIDE
        or height > config.MAX_IMAGE_SIDE
    ):
        raise ValueError(
            "The screenshot is %d by %d pixels. The limit is %d pixels in total and %d per side."
            % (width, height, config.MAX_IMAGE_PIXELS, config.MAX_IMAGE_SIDE)
        )
    try:
        # Decode fully now so a corrupt file is reported as a validation error.
        image = Image.open(io.BytesIO(data))
        image.load()
        rgb = image.convert("RGB")
    except Exception:  # noqa: BLE001 - Pillow raises many types for corrupt data
        raise ValueError("The screenshot is corrupt and could not be decoded.") from None
    info = {"format": fmt, "width": width, "height": height, "bytes": len(data)}
    return rgb, info


def validate_case(
    url: Optional[str] = None,
    message_text: Optional[str] = None,
    screenshot_bytes: Optional[bytes] = None,
) -> CaseInput:
    """Validate the three optional parts and build a CaseInput.

    Blank strings and empty uploads count as not provided.
    """
    errors: List[dict] = []
    case = CaseInput()

    if not _blank(url):
        try:
            case.url = validate_url(url)
        except ValueError as exc:
            errors.append({"field": "url", "message": str(exc)})

    if not _blank(message_text):
        try:
            case.message_text = validate_text(message_text)
        except ValueError as exc:
            errors.append({"field": "message_text", "message": str(exc)})

    if screenshot_bytes:
        try:
            case.screenshot, case.screenshot_info = validate_image(screenshot_bytes)
        except ValueError as exc:
            errors.append({"field": "screenshot", "message": str(exc)})

    if not errors and not any(case.provided().values()):
        errors.append(
            {
                "field": "case",
                "message": "Provide at least one of: a URL, a message text, or a screenshot.",
            }
        )
    if errors:
        raise ValidationError(errors)
    return case
