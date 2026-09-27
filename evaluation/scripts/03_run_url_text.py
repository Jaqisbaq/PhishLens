"""Part A step 3 (url, text branches): run the pinned URL and text models on
every case in the linked case table and cache P(phishing) plus token counts
and truncation flags to disk, keyed by case id. Resumable: already-cached
ids are skipped.
"""
from __future__ import annotations

import sys
import time

import pandas as pd

import common

common.configure_hf_env()
common.ensure_src_on_path()

CASES = common.CACHE_DIR / "case_table.parquet"
OUT_URL = common.CACHE_DIR / "url_predictions.parquet"
OUT_TEXT = common.CACHE_DIR / "text_predictions.parquet"


def run_url(cases: pd.DataFrame) -> None:
    from phishlens.adapters.url_model import UrlModelAdapter

    done = set()
    if OUT_URL.exists():
        done = set(pd.read_parquet(OUT_URL)["id"])
    todo = cases[~cases["id"].isin(done)]
    print(f"[url] {len(done)} cached, {len(todo)} to run")
    if todo.empty:
        return
    adapter = UrlModelAdapter()
    adapter.load()
    t0 = time.time()
    probs = adapter.predict_batch(list(todo["url"]), batch_size=64)
    tokens = [adapter.count_tokens(u) for u in todo["url"]]
    elapsed = time.time() - t0
    print(f"[url] {len(todo)} items in {elapsed:.1f}s ({elapsed / len(todo) * 1000:.1f} ms/item)")
    new = pd.DataFrame({
        "id": todo["id"].values,
        "p_url": probs,
        "url_token_count": tokens,
        "url_truncated": [t > adapter.max_tokens for t in tokens],
    })
    out = pd.concat([pd.read_parquet(OUT_URL), new], ignore_index=True) if OUT_URL.exists() else new
    out.to_parquet(OUT_URL, index=False)
    print("[url] wrote", OUT_URL, "total rows", len(out))


def run_text(cases: pd.DataFrame) -> None:
    from phishlens.adapters.text_model import TextModelAdapter

    done = set()
    if OUT_TEXT.exists():
        done = set(pd.read_parquet(OUT_TEXT)["id"])
    todo = cases[~cases["id"].isin(done)]
    print(f"[text] {len(done)} cached, {len(todo)} to run")
    if todo.empty:
        return
    adapter = TextModelAdapter()
    adapter.load()
    texts = list(todo["text"])
    ids = list(todo["id"])
    # Process and checkpoint in chunks so a long run is resumable if interrupted.
    chunk = 25
    rows = []
    t_start = time.time()
    for start in range(0, len(texts), chunk):
        batch_ids = ids[start:start + chunk]
        batch_texts = texts[start:start + chunk]
        t0 = time.time()
        probs = adapter.predict_batch(batch_texts, batch_size=8)
        tok = [adapter.count_tokens(t) for t in batch_texts]
        dt = time.time() - t0
        for i, p, tk in zip(batch_ids, probs, tok):
            rows.append({"id": i, "p_text": p, "text_token_count": tk,
                         "text_truncated": tk > adapter.max_tokens})
        done_n = start + len(batch_ids)
        elapsed = time.time() - t_start
        rate = elapsed / done_n
        remaining = (len(texts) - done_n) * rate
        print(f"[text] {done_n}/{len(texts)} chunk={dt:.1f}s "
              f"avg={rate*1000:.0f}ms/item eta={remaining/60:.1f}min", flush=True)
        # checkpoint
        new = pd.DataFrame(rows)
        out = pd.concat([pd.read_parquet(OUT_TEXT), new], ignore_index=True) if OUT_TEXT.exists() else new
        out.to_parquet(OUT_TEXT, index=False)
        rows = []
    print("[text] done")


def main() -> None:
    cases = pd.read_parquet(CASES)
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("url", "both"):
        run_url(cases)
    if which in ("text", "both"):
        run_text(cases)


if __name__ == "__main__":
    main()
