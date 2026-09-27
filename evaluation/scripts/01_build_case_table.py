"""Part A step 1: build the linked case table and remove training-data leakage.

Reads the usable linked triples (URL + page text + non-blank screenshot) from
`zenodo_8041387_linked_check.csv` (three == True), pulls the page text from
phishing.csv / not-phishing.csv by _id, removes any case whose URL matches
(exact or normalised) an entry in ealvaradob/phishing-dataset urls.json (the
text and URL models' training data), and writes the resulting case table plus
a small filtering-step count table.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

import common

LINKED_CHECK = common.DATA_LOGS / "zenodo_8041387_linked_check.csv"
PHISHING_CSV = common.DATA_RAW / "zenodo_8041387_phishing_website" / "phishing.csv"
NOTPHISH_CSV = common.DATA_RAW / "zenodo_8041387_phishing_website" / "not-phishing.csv"
SCREEN_DIR = common.DATA_RAW / "zenodo_8041387_phishing_website" / "screenshots"
TRAIN_URLS = common.DATA_RAW / "ealvaradob_phishing_dataset" / "urls.json"

OUT_CASES = common.CACHE_DIR / "case_table.parquet"
OUT_STEPS = common.TABLES_DIR / "dataset_summary.csv"


def normalise_url(u: str) -> str:
    if not isinstance(u, str) or not u:
        return ""
    u = u.strip().lower()
    try:
        parts = urlsplit(u if "://" in u else "http://" + u)
    except ValueError:
        return u.rstrip("/")
    host = (parts.hostname or "").lstrip("www.")
    path = parts.path.rstrip("/")
    return host + path


def load_text_map(path: Path) -> dict:
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["_id"]] = row.get("features.text", "") or ""
    return out


def main() -> None:
    linked = pd.read_csv(LINKED_CHECK)
    usable = linked[linked["three"] == True].copy()  # noqa: E712
    assert len(usable) == 1022, f"expected 1022 usable triples, got {len(usable)}"
    n_phish_raw = int((usable["label_csv"] == 1).sum())
    n_legit_raw = int((usable["label_csv"] == 0).sum())

    text_phish = load_text_map(PHISHING_CSV)
    text_legit = load_text_map(NOTPHISH_CSV)

    rows = []
    for _, r in usable.iterrows():
        cid = r["_id"]
        label = int(r["label_csv"])
        text = text_phish.get(cid) if label == 1 else text_legit.get(cid)
        subdir = "phishing" if label == 1 else "not-phishing"
        shot = SCREEN_DIR / subdir / f"{cid}.jpg"
        rows.append({
            "id": cid,
            "label": label,
            "url": r["url"],
            "url_norm": normalise_url(r["url"]),
            "text": text or "",
            "domain": r["domain"],
            "language": r.get("language", ""),
            "screenshot_path": str(shot.relative_to(common.WORK_DIR)),
            "screenshot_exists": shot.exists(),
        })
    cases = pd.DataFrame(rows)
    assert cases["screenshot_exists"].all(), "some usable triples are missing their screenshot file"
    assert cases["text"].str.len().gt(0).all(), "some usable triples have empty page text"

    # Leakage removal against ealvaradob/phishing-dataset urls.json (training
    # data for both the URL and the text model).
    with open(TRAIN_URLS, encoding="utf-8") as f:
        train_urls_raw = json.load(f)
    if isinstance(train_urls_raw, dict):
        train_url_list = list(train_urls_raw.get("url", train_urls_raw.get("urls", [])))
        if not train_url_list:
            # fall back: values of the dict are the urls
            train_url_list = [v for v in train_urls_raw.values() if isinstance(v, str)]
    else:
        train_url_list = train_urls_raw
    exact_set = set()
    norm_set = set()
    for item in train_url_list:
        if isinstance(item, dict):
            u = item.get("url") or item.get("text") or ""
        else:
            u = item
        if not isinstance(u, str) or not u:
            continue
        exact_set.add(u.strip().lower())
        norm_set.add(normalise_url(u))

    cases["url_lower"] = cases["url"].str.strip().str.lower()
    cases["leak_exact"] = cases["url_lower"].isin(exact_set)
    cases["leak_norm"] = cases["url_norm"].isin(norm_set)
    cases["is_leak"] = cases["leak_exact"] | cases["leak_norm"]

    n_leak_phish = int(cases[(cases.label == 1) & cases.is_leak].shape[0])
    n_leak_legit = int(cases[(cases.label == 0) & cases.is_leak].shape[0])

    clean = cases[~cases.is_leak].drop(
        columns=["url_lower", "leak_exact", "leak_norm", "is_leak", "screenshot_exists"]
    ).reset_index(drop=True)

    n_phish_clean = int((clean.label == 1).sum())
    n_legit_clean = int((clean.label == 0).sum())

    clean.to_parquet(OUT_CASES, index=False)

    any_present = linked[linked["three_any"] == True]  # noqa: E712
    n_any_phish = int((any_present["label_csv"] == 1).sum())
    n_any_legit = int((any_present["label_csv"] == 0).sum())
    steps = pd.DataFrame([
        {"step": "screenshot_present_any (three_any)", "phishing": n_any_phish,
         "legitimate": n_any_legit},
        {"step": "usable_triple (three: present, not blank)", "phishing": n_phish_raw, "legitimate": n_legit_raw},
        {"step": "removed_training_leakage (url exact or normalised match vs ealvaradob urls.json)",
         "phishing": n_leak_phish, "legitimate": n_leak_legit},
        {"step": "final_linked_case_table", "phishing": n_phish_clean, "legitimate": n_legit_clean},
    ])
    steps.to_csv(OUT_STEPS, index=False)

    print("usable triples:", n_phish_raw, "phishing /", n_legit_raw, "legitimate")
    print("leakage removed:", n_leak_phish, "phishing /", n_leak_legit, "legitimate")
    print("final case table:", n_phish_clean, "phishing /", n_legit_clean, "legitimate, total", len(clean))
    print("wrote", OUT_CASES)
    print("wrote", OUT_STEPS)


if __name__ == "__main__":
    main()
