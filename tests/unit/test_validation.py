"""Validation: accept and reject cases, scheme preserved, image checks."""
from __future__ import annotations

import io

import pytest
from PIL import Image

from phishlens import config
from phishlens.validation import ValidationError, validate_case, validate_url


def _png_bytes(width=10, height=10) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


# ---- URL ---------------------------------------------------------------
def test_valid_https_url_scheme_preserved():
    out = validate_url("https://example.com/a/b?x=1")
    assert out == "https://example.com/a/b?x=1"


def test_valid_http_url_scheme_preserved():
    out = validate_url("http://192.0.2.5/status")
    assert out == "http://192.0.2.5/status"


def test_url_strips_only_surrounding_whitespace():
    out = validate_url("  https://example.org/path  ")
    assert out == "https://example.org/path"


def test_url_missing_scheme_rejected():
    with pytest.raises(ValueError):
        validate_url("example.com/login")


def test_url_bad_scheme_rejected():
    with pytest.raises(ValueError):
        validate_url("ftp://example.com/file")


def test_url_no_host_rejected():
    with pytest.raises(ValueError):
        validate_url("https:///path")


def test_url_too_long_rejected():
    long_url = "https://example.com/" + "a" * config.MAX_URL_LENGTH
    with pytest.raises(ValueError):
        validate_url(long_url)


def test_url_with_space_rejected():
    with pytest.raises(ValueError):
        validate_url("https://example.com/a b")


# ---- text ----------------------------------------------------------------
def test_text_too_long_rejected():
    with pytest.raises(ValidationError):
        validate_case(message_text="a" * (config.MAX_TEXT_LENGTH + 1))


def test_text_accepted_and_stripped():
    case = validate_case(message_text="  hello there  ")
    assert case.message_text == "hello there"


# ---- image -----------------------------------------------------------------
def test_image_valid_png_accepted():
    case = validate_case(screenshot_bytes=_png_bytes())
    assert case.screenshot is not None
    assert case.screenshot_info["format"] == "PNG"


def test_image_too_large_rejected():
    with pytest.raises(ValidationError):
        validate_case(screenshot_bytes=b"x" * (config.MAX_IMAGE_BYTES + 1))


def test_image_corrupt_rejected():
    with pytest.raises(ValidationError):
        validate_case(screenshot_bytes=b"not an image at all")


def test_image_wrong_format_rejected():
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), "white").save(buf, format="BMP")
    with pytest.raises(ValidationError):
        validate_case(screenshot_bytes=buf.getvalue())


def test_image_decompression_bomb_guard_rejected():
    # Larger than MAX_IMAGE_PIXELS but still small on disk (uniform colour PNG).
    huge = Image.new("RGB", (6000, 6000), "white")
    buf = io.BytesIO()
    huge.save(buf, format="PNG")
    data = buf.getvalue()
    if len(data) > config.MAX_IMAGE_BYTES:
        pytest.skip("fixture image exceeds the byte limit before the pixel check runs")
    with pytest.raises(ValidationError):
        validate_case(screenshot_bytes=data)


# ---- whole case --------------------------------------------------------
def test_case_requires_at_least_one_field():
    with pytest.raises(ValidationError) as exc_info:
        validate_case()
    assert any(e["field"] == "case" for e in exc_info.value.errors)


def test_case_accepts_url_only():
    case = validate_case(url="https://example.com/")
    assert case.provided() == {"url": True, "text": False, "image": False}


def test_case_accepts_all_three():
    case = validate_case(
        url="https://example.com/", message_text="hello", screenshot_bytes=_png_bytes()
    )
    assert case.provided() == {"url": True, "text": True, "image": True}


def test_case_blank_strings_count_as_not_provided():
    with pytest.raises(ValidationError) as exc_info:
        validate_case(url="   ", message_text="")
    assert any(e["field"] == "case" for e in exc_info.value.errors)


def test_case_collects_multiple_errors():
    with pytest.raises(ValidationError) as exc_info:
        validate_case(url="not-a-url", message_text="a" * (config.MAX_TEXT_LENGTH + 1))
    fields = {e["field"] for e in exc_info.value.errors}
    assert fields == {"url", "message_text"}
