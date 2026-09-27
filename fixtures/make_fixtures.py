"""Build the synthetic fixtures used by tests and screenshots.

Everything here is harmless and synthetic: hosts on example.com/.org/.net
and the 192.0.2.0/24 documentation range, invented messages, and
screenshots drawn with Pillow (never a real captured screenshot).

Run once: python fixtures/make_fixtures.py
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
IMAGES_DIR = HERE / "images"


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def draw_login_page() -> Image.Image:
    """A synthetic image resembling a login page asking for a password."""
    img = Image.new("RGB", (800, 600), "#f2f2f2")
    d = ImageDraw.Draw(img)
    d.rectangle([250, 120, 550, 480], fill="#ffffff", outline="#cccccc")
    d.text((280, 150), "Account Sign In", fill="black", font=_font(24))
    d.text((280, 210), "Username", fill="black", font=_font(16))
    d.rectangle([280, 235, 520, 265], outline="#888888")
    d.text((280, 290), "Password", fill="black", font=_font(16))
    d.rectangle([280, 315, 520, 345], outline="#888888")
    d.rectangle([280, 380, 520, 415], fill="#3366cc")
    d.text((350, 388), "Sign In", fill="white", font=_font(16))
    d.text((280, 440), "Your account will be suspended unless you confirm now.",
           fill="#cc0000", font=_font(13))
    return img


def draw_news_article() -> Image.Image:
    """A synthetic image resembling an ordinary news article page."""
    img = Image.new("RGB", (800, 600), "#ffffff")
    d = ImageDraw.Draw(img)
    d.text((40, 30), "Example Daily News", fill="black", font=_font(26))
    d.line([40, 70, 760, 70], fill="#dddddd")
    d.text((40, 100), "Local council approves new park budget", fill="black", font=_font(18))
    lines = [
        "The council met on Tuesday to review the annual budget for park",
        "maintenance. Officials said the plan would fund new benches and",
        "walking paths across three neighbourhoods over the next year.",
        "A public comment session is scheduled for next month.",
    ]
    y = 140
    for line in lines:
        d.text((40, y), line, fill="#333333", font=_font(14))
        y += 24
    return img


def draw_blank_table() -> Image.Image:
    """A synthetic image resembling a plain table of information."""
    img = Image.new("RGB", (800, 600), "#ffffff")
    d = ImageDraw.Draw(img)
    d.text((40, 30), "Quarterly Report", fill="black", font=_font(22))
    headers = ["Quarter", "Units", "Notes"]
    rows = [["Q1", "120", "steady"], ["Q2", "134", "up"], ["Q3", "128", "flat"]]
    col_x = [40, 220, 400]
    y = 90
    for x, h in zip(col_x, headers):
        d.text((x, y), h, fill="black", font=_font(16))
    y += 30
    for row in rows:
        for x, val in zip(col_x, row):
            d.text((x, y), val, fill="#333333", font=_font(14))
        y += 28
    return img


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    images = {
        "login_page.png": draw_login_page(),
        "news_article.png": draw_news_article(),
        "table.png": draw_blank_table(),
    }
    for name, img in images.items():
        img.save(IMAGES_DIR / name)

    cases = {
        "urls": {
            "benign_plain": "https://example.com/account/login",
            "benign_org": "https://example.org/support/help",
            "benign_ip_doc_range": "http://192.0.2.10/status",
            "suspicious_shape": "http://192.0.2.44/secure-login-update@example.net/confirm-account-details",
        },
        "messages": {
            "benign": "Hi team, the meeting notes from example.org are attached. See you Monday.",
            "urgent_style": (
                "URGENT: your account on example.net will be suspended in 24 hours. "
                "Confirm your password and card details now at the link below to keep access."
            ),
        },
        "images": {
            "login_page": "images/login_page.png",
            "news_article": "images/news_article.png",
            "table": "images/table.png",
        },
    }
    with open(HERE / "cases.json", "w", encoding="utf-8") as handle:
        json.dump(cases, handle, indent=2)
    print("Wrote %d images and fixtures/cases.json" % len(images))


if __name__ == "__main__":
    main()
