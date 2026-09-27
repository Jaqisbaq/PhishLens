"""Part B: URL branch on an independent sample of PhiUSIIL, after leakage
removal against ealvaradob/phishing-dataset urls.json (the URL model's
training data). Stratified random sample of 20,000, seed 2026.
"""
from __future__ import annotations

import json
import zipfile
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

import common
common.configure_hf_env()
common.ensure_src_on_path()

ZIP_PATH = common.DATA_RAW / "uci_phiusiil" / "phiusiil_phishing_url_dataset.zip"
TRAIN_URLS = common.DATA_RAW / "ealvaradob_phishing_dataset" / "urls.json"
OUT_SAMPLE = common.CACHE_DIR / "partb_url_sample.parquet"
OUT_PREDS = common.CACHE_DIR / "partb_url_predictions.parquet"
OUT_STEPS = common.TABLES_DIR / "partb_url_filtering.csv"

SAMPLE_N = 20000
SEED = common.SEED


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


def main() -> None:
    with zipfile.ZipFile(ZIP_PATH) as z:
        with z.open("PhiUSIIL_Phishing_URL_Dataset.csv") as f:
            df = pd.read_csv(f, usecols=["URL", "label"])
    df = df.rename(columns={"URL": "url"})
    # PhiUSIIL label: 1 legitimate, 0 phishing -> convert to phishing=1 convention
    df["label"] = 1 - df["label"]
    n_raw_phish = int((df.label == 1).sum())
    n_raw_legit = int((df.label == 0).sum())

    with open(TRAIN_URLS, encoding="utf-8") as f:
        train_urls_raw = json.load(f)
    train_url_list = train_urls_raw if isinstance(train_urls_raw, list) else list(train_urls_raw.values())
    exact_set, norm_set = set(), set()
    for item in train_url_list:
        u = (item.get("url") or item.get("text")) if isinstance(item, dict) else item
        if isinstance(u, str) and u:
            exact_set.add(u.strip().lower())
            norm_set.add(normalise_url(u))

    df["url_lower"] = df["url"].str.strip().str.lower()
    df["url_norm"] = df["url"].map(normalise_url)
    df["is_leak"] = df["url_lower"].isin(exact_set) | df["url_norm"].isin(norm_set)
    n_leak_phish = int(df[(df.label == 1) & df.is_leak].shape[0])
    n_leak_legit = int(df[(df.label == 0) & df.is_leak].shape[0])

    clean = df[~df.is_leak].drop(columns=["url_lower", "url_norm", "is_leak"]).reset_index(drop=True)

    rng = np.random.RandomState(SEED)
    frac_phish = SAMPLE_N * (n_raw_phish / (n_raw_phish + n_raw_legit))
    n_phish_sample = int(round(frac_phish))
    n_legit_sample = SAMPLE_N - n_phish_sample

    phish_pool = clean[clean.label == 1]
    legit_pool = clean[clean.label == 0]
    phish_sample = phish_pool.sample(n=min(n_phish_sample, len(phish_pool)), random_state=SEED)
    legit_sample = legit_pool.sample(n=min(n_legit_sample, len(legit_pool)), random_state=SEED)
    sample = pd.concat([phish_sample, legit_sample], ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    sample["id"] = ["phiusiil_%06d" % i for i in range(len(sample))]

    sample.to_parquet(OUT_SAMPLE, index=False)

    steps = pd.DataFrame([
        {"step": "phiusiil_raw", "phishing": n_raw_phish, "legitimate": n_raw_legit},
        {"step": "removed_training_leakage", "phishing": n_leak_phish, "legitimate": n_leak_legit},
        {"step": "clean_pool", "phishing": len(phish_pool), "legitimate": len(legit_pool)},
        {"step": "stratified_sample", "phishing": int((sample.label == 1).sum()),
         "legitimate": int((sample.label == 0).sum())},
    ])
    steps.to_csv(OUT_STEPS, index=False)
    print(steps)

    # Run the URL model.
    from phishlens.adapters.url_model import UrlModelAdapter
    adapter = UrlModelAdapter()
    adapter.load()
    probs = adapter.predict_batch(list(sample["url"]), batch_size=128)
    pd.DataFrame({"id": sample["id"], "label": sample["label"], "url": sample["url"],
                  "p_url": probs}).to_parquet(OUT_PREDS, index=False)
    print("wrote", OUT_PREDS)


if __name__ == "__main__":
    main()
