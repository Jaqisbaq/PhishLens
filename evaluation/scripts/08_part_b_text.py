"""Part B: text branch on an independent sample of CEAS_08 (email text, 0
percent overlap with the text model's training data per docs/data/DATA.md).
500 phishing and 500 legitimate, seed 2026. Nazario phishing-only rows are
not needed because CEAS_08 has enough of both classes.
"""
from __future__ import annotations

import pandas as pd

import common
common.configure_hf_env()
common.ensure_src_on_path()

CEAS = common.DATA_RAW / "zenodo_8339691_email_curated" / "CEAS_08.csv"
OUT_SAMPLE = common.CACHE_DIR / "partb_text_sample.parquet"
OUT_PREDS = common.CACHE_DIR / "partb_text_predictions.parquet"

N_PER_CLASS = 500
SEED = common.SEED


def main() -> None:
    df = pd.read_csv(CEAS)
    df["body"] = df["body"].fillna("")
    df["subject"] = df["subject"].fillna("")
    df["text"] = "Subject: " + df["subject"] + "\n\n" + df["body"]
    df = df[df["text"].str.strip().str.len() > 0]

    phish = df[df.label == 1].sample(n=N_PER_CLASS, random_state=SEED)
    legit = df[df.label == 0].sample(n=N_PER_CLASS, random_state=SEED)
    sample = pd.concat([phish, legit], ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    sample["id"] = ["ceas08_%04d" % i for i in range(len(sample))]
    sample[["id", "label", "text"]].to_parquet(OUT_SAMPLE, index=False)
    print("sample:", int((sample.label == 1).sum()), "phishing,",
          int((sample.label == 0).sum()), "legitimate")

    from phishlens.adapters.text_model import TextModelAdapter
    adapter = TextModelAdapter()
    adapter.load()

    import time
    texts = list(sample["text"])
    ids = list(sample["id"])
    rows = []
    t0 = time.time()
    chunk = 25
    for start in range(0, len(texts), chunk):
        bt = texts[start:start + chunk]
        bi = ids[start:start + chunk]
        probs = adapter.predict_batch(bt, batch_size=8)
        tok = [adapter.count_tokens(t) for t in bt]
        for i, p, tk in zip(bi, probs, tok):
            rows.append({"id": i, "p_text": p, "token_count": tk,
                         "truncated": tk > adapter.max_tokens})
        done = start + len(bt)
        elapsed = time.time() - t0
        print(f"[partb text] {done}/{len(texts)} eta={((len(texts)-done)*elapsed/done)/60:.1f}min",
              flush=True)
    preds = pd.DataFrame(rows)
    out = sample.merge(preds, on="id")
    out.to_parquet(OUT_PREDS, index=False)
    print("wrote", OUT_PREDS)


if __name__ == "__main__":
    main()
