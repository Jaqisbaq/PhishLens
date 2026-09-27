"""Regression test: every outcome label must have a matching CSS rule.

The interface derives the badge class from the outcome label in app.js. An
earlier version joined the words with a dot, which produced a class name
that no CSS selector could match, so the badge was shown without colour.
"""
import re
from pathlib import Path

import pytest

from phishlens import schemas

STATIC_DIR = Path(schemas.__file__).resolve().parent / "static"

OUTCOMES = [
    schemas.OUTCOME_HIGH,
    schemas.OUTCOME_REVIEW,
    schemas.OUTCOME_LIMITED,
    schemas.OUTCOME_UNAVAILABLE,
]


def css_class_for(outcome: str) -> str:
    """Mirror of outcomeClass() in app.js."""
    return "outcome-" + re.sub(r"\s+", "-", outcome.lower())


@pytest.mark.parametrize("outcome", OUTCOMES)
def test_outcome_has_single_class_css_rule(outcome):
    css = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
    cls = css_class_for(outcome)
    assert re.search(r"\." + re.escape(cls) + r"\s*\{", css), cls


@pytest.mark.parametrize("outcome", OUTCOMES)
def test_outcome_class_is_a_valid_single_token(outcome):
    cls = css_class_for(outcome)
    assert re.fullmatch(r"[a-z][a-z0-9-]*", cls), cls


def test_app_js_uses_the_same_class_rule():
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert 'outcome.toLowerCase().replace(/\\s+/g, "-")' in js
