"""Part B: candidate comparison. On a 400 item validation sample per modality,
drawn separately from the Part B samples, run the rejected candidate models
alongside the chosen primary, and record training-data overlap where known.

URL candidates: pirocheto/phishing-url-detection (ONNX), kmack/malicious-url-detection.
Text candidates: aamoshdahal/email-phishing-distilbert-finetuned (contaminated:
trained on CEAS 2008 and Nazario, same source as this sample), ElSlay/BERT-Phishing-Email-Model.
cybersectony and DomURLs_BERT are excluded (unnamed four-class output; no classifier), per spec.
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
CEAS = common.DATA_RAW / "zenodo_8339691_email_curated" / "CEAS_08.csv"
TRAIN_URLS = common.DATA_RAW / "ealvaradob_phishing_dataset" / "urls.json"

PARTB_URL_SAMPLE = common.CACHE_DIR / "partb_url_sample.parquet"
PARTB_TEXT_SAMPLE = common.CACHE_DIR / "partb_text_sample.parquet"

OUT_URL_SAMPLE = common.CACHE_DIR / "candidates_url_sample.parquet"
OUT_TEXT_SAMPLE = common.CACHE_DIR / "candidates_text_sample.parquet"
OUT_TABLE = common.TABLES_DIR / "candidate_comparison.csv"

N_VAL = 400
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
    return host + parts.path.rstrip("/")


def build_url_sample():
    with zipfile.ZipFile(ZIP_PATH) as z:
        with z.open("PhiUSIIL_Phishing_URL_Dataset.csv") as f:
            df = pd.read_csv(f, usecols=["URL", "label"])
    df = df.rename(columns={"URL": "url"})
    df["label"] = 1 - df["label"]  # 1 phishing, 0 legitimate

    with open(TRAIN_URLS, encoding="utf-8") as f:
        train_list = json.load(f)
    exact_set, norm_set = set(), set()
    for item in train_list:
        u = item.get("text") if isinstance(item, dict) else item
        if isinstance(u, str) and u:
            exact_set.add(u.strip().lower())
            norm_set.add(normalise_url(u))
    df["url_lower"] = df["url"].str.strip().str.lower()
    df["url_norm"] = df["url"].map(normalise_url)
    df["is_leak"] = df["url_lower"].isin(exact_set) | df["url_norm"].isin(norm_set)
    clean = df[~df.is_leak].reset_index(drop=True)

    used = set(pd.read_parquet(PARTB_URL_SAMPLE)["url"]) if PARTB_URL_SAMPLE.exists() else set()
    remaining = clean[~clean["url"].isin(used)]

    n_phish = N_VAL // 2
    n_legit = N_VAL - n_phish
    phish = remaining[remaining.label == 1].sample(n=n_phish, random_state=SEED)
    legit = remaining[remaining.label == 0].sample(n=n_legit, random_state=SEED)
    sample = pd.concat([phish, legit], ignore_index=True).sample(frac=1.0, random_state=SEED)
    sample = sample.reset_index(drop=True)
    sample["id"] = ["candurl_%04d" % i for i in range(len(sample))]
    sample[["id", "url", "label"]].to_parquet(OUT_URL_SAMPLE, index=False)
    return sample[["id", "url", "label"]]


def build_text_sample():
    df = pd.read_csv(CEAS)
    df["body"] = df["body"].fillna("")
    df["subject"] = df["subject"].fillna("")
    df["text"] = "Subject: " + df["subject"] + "\n\n" + df["body"]
    df = df[df["text"].str.strip().str.len() > 0]

    used = set(pd.read_parquet(PARTB_TEXT_SAMPLE)["text"]) if PARTB_TEXT_SAMPLE.exists() else set()
    remaining = df[~df["text"].isin(used)]

    n_phish = N_VAL // 2
    n_legit = N_VAL - n_phish
    phish = remaining[remaining.label == 1].sample(n=n_phish, random_state=SEED)
    legit = remaining[remaining.label == 0].sample(n=n_legit, random_state=SEED)
    sample = pd.concat([phish, legit], ignore_index=True).sample(frac=1.0, random_state=SEED)
    sample = sample.reset_index(drop=True)
    sample["id"] = ["candtext_%04d" % i for i in range(len(sample))]
    sample[["id", "text", "label"]].to_parquet(OUT_TEXT_SAMPLE, index=False)
    return sample[["id", "text", "label"]]


def run_seqcls(repo, rev, texts, max_len, phishing_index):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(repo, revision=rev)
    model.eval()
    probs = []
    bs = 16
    for start in range(0, len(texts), bs):
        batch = texts[start:start + bs]
        enc = tok(batch, return_tensors="pt", truncation=True, max_length=max_len, padding=True)
        with torch.no_grad():
            logits = model(**enc).logits
        p = torch.softmax(logits, dim=-1)[:, phishing_index]
        probs.extend([float(x) for x in p.tolist()])
    return probs


def run_kmack(urls):
    return run_seqcls("kmack/malicious-url-detection",
                       "258499831602e1aea6c1f00e8483b820dd14b391", urls, 128, 1)


def run_pirocheto(urls):
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("pirocheto/phishing-url-detection", "model.onnx",
                            revision="44f3b19f705b52532e0aadf3d0d15dd892b8a2fb")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    probs = []
    for u in urls:
        out = sess.run(None, {"inputs": np.array([u], dtype="str")})
        proba = np.asarray(out[1])[0]
        probs.append(float(proba[1]))
    return probs


def run_aamoshdahal(texts):
    return run_seqcls("aamoshdahal/email-phishing-distilbert-finetuned",
                       "f9d211125fcb6e85cf73630957667bc2a6e93ed3", texts, 512, 1)


def run_elslay(texts):
    return run_seqcls("ElSlay/BERT-Phishing-Email-Model",
                       "708d260869707ffdb0227c3c838ce957a736c9d6", texts, 512, 1)


def score_metrics(y, p):
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
    pred = (np.asarray(p) >= 0.5).astype(int)
    out = {
        "n": len(y),
        "average_precision": float(average_precision_score(y, p)),
        "f1_at_0.5": float(f1_score(y, pred, zero_division=0)),
    }
    out["roc_auc"] = float(roc_auc_score(y, p)) if len(set(y)) > 1 else None
    return out


def main() -> None:
    url_sample = build_url_sample() if not OUT_URL_SAMPLE.exists() else pd.read_parquet(OUT_URL_SAMPLE)
    text_sample = build_text_sample() if not OUT_TEXT_SAMPLE.exists() else pd.read_parquet(OUT_TEXT_SAMPLE)

    rows = []

    # --- URL modality ---
    from phishlens.adapters.url_model import UrlModelAdapter
    primary = UrlModelAdapter()
    primary.load()
    p_primary = primary.predict_batch(list(url_sample["url"]), batch_size=64)
    m = score_metrics(url_sample["label"], p_primary)
    rows.append({"modality": "url", "model": "CrabInHoney/urlbert-tiny-v4-phishing-classifier (primary)",
                 "contaminated_on_this_sample": False, **m})

    p_kmack = run_kmack(list(url_sample["url"]))
    m = score_metrics(url_sample["label"], p_kmack)
    rows.append({"modality": "url", "model": "kmack/malicious-url-detection (rejected: labels "
                 "BENIGN/MALWARE, not phishing-specific)", "contaminated_on_this_sample": False, **m})

    p_piro = run_pirocheto(list(url_sample["url"]))
    m = score_metrics(url_sample["label"], p_piro)
    rows.append({"modality": "url", "model": "pirocheto/phishing-url-detection (rejected: weak on "
                 "benign samples in the spike; trained on pirocheto/phishing-url, no overlap with "
                 "PhiUSIIL known)", "contaminated_on_this_sample": False, **m})

    # --- text modality ---
    from phishlens.adapters.text_model import TextModelAdapter
    primary_t = TextModelAdapter()
    primary_t.load()
    p_primary_t = primary_t.predict_batch(list(text_sample["text"]), batch_size=8)
    m = score_metrics(text_sample["label"], p_primary_t)
    rows.append({"modality": "text", "model": "ealvaradob/bert-finetuned-phishing (primary)",
                 "contaminated_on_this_sample": False, **m})

    p_aamosh = run_aamoshdahal(list(text_sample["text"]))
    m = score_metrics(text_sample["label"], p_aamosh)
    rows.append({"modality": "text", "model": "aamoshdahal/email-phishing-distilbert-finetuned "
                 "(rejected: no licence stated)",
                 "contaminated_on_this_sample": True,
                 "contamination_note": "trained on Enron, CEAS 2008, Ling-Spam, SpamAssassin, "
                 "Nazario and Nigerian fraud emails per its card, which includes CEAS_08: this "
                 "sample partially overlaps its own training data, so its score here is optimistic "
                 "and not comparable to the other rows",
                 **m})

    p_elslay = run_elslay(list(text_sample["text"]))
    m = score_metrics(text_sample["label"], p_elslay)
    rows.append({"modality": "text", "model": "ElSlay/BERT-Phishing-Email-Model (rejected: no "
                 "licence stated, higher latency, no accuracy gain in the spike)",
                 "contaminated_on_this_sample": False, **m})

    rows.append({"modality": "url", "model": "amahdaouy/DomURLs_BERT",
                 "excluded_reason": "no classifier head in the checkpoint (encoder only); would "
                 "need a labelled URL set and a fitted head before it produces any score"})
    rows.append({"modality": "text", "model": "cybersectony/phishing-email-detection-distilbert_v2.4.1",
                 "excluded_reason": "four-class output with no class names in the config or "
                 "agreement between the model card and dataset card on what the classes mean"})

    pd.DataFrame(rows).to_csv(OUT_TABLE, index=False)
    print(pd.DataFrame(rows))
    print("wrote", OUT_TABLE)


if __name__ == "__main__":
    main()
