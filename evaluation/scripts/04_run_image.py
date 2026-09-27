"""Part A step 3/4 (image branch): compute CLIP embeddings and the zero-shot
score for every case's screenshot, cached to disk keyed by case id.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from PIL import Image

import common

common.configure_hf_env()
common.ensure_src_on_path()

CASES = common.CACHE_DIR / "case_table.parquet"
OUT_EMB = common.CACHE_DIR / "image_embeddings.npz"
OUT_ZS = common.CACHE_DIR / "image_zeroshot.parquet"


def main() -> None:
    from phishlens.adapters.image_model import ImageModelAdapter

    cases = pd.read_parquet(CASES)
    ids = list(cases["id"])

    done_ids = set()
    if OUT_EMB.exists():
        npz = np.load(OUT_EMB, allow_pickle=True)
        done_ids = set(npz["ids"].tolist())

    todo = cases[~cases["id"].isin(done_ids)]
    print(f"[image] {len(done_ids)} cached, {len(todo)} to run")

    adapter = ImageModelAdapter()
    adapter.load()

    if not todo.empty:
        images = []
        good_ids = []
        for _, r in todo.iterrows():
            path = common.WORK_DIR / r["screenshot_path"]
            img = Image.open(path)
            img.load()
            images.append(img.convert("RGB"))
            good_ids.append(r["id"])
        t0 = time.time()
        embeddings = adapter.embed_images(images, batch_size=16)
        elapsed = time.time() - t0
        print(f"[image] embedded {len(good_ids)} in {elapsed:.1f}s "
              f"({elapsed/len(good_ids)*1000:.1f} ms/item)")

        zs_rows = []
        for i, emb in zip(good_ids, embeddings):
            scored = adapter.score_embedding(emb, mode="zero_shot")
            zs_rows.append({
                "id": i,
                "p_image_zero_shot": scored["probability"],
                "top_prompt": scored["top_prompt"],
                "top_prompt_score": scored["top_prompt_score"],
                "top_prompt_group": scored["top_prompt_group"],
            })

        if OUT_EMB.exists():
            npz = np.load(OUT_EMB, allow_pickle=True)
            all_ids = np.concatenate([npz["ids"], np.array(good_ids, dtype=object)])
            all_emb = np.concatenate([npz["embeddings"], embeddings], axis=0)
        else:
            all_ids = np.array(good_ids, dtype=object)
            all_emb = embeddings
        np.savez(OUT_EMB, ids=all_ids, embeddings=all_emb)

        new_zs = pd.DataFrame(zs_rows)
        out_zs = pd.concat([pd.read_parquet(OUT_ZS), new_zs], ignore_index=True) if OUT_ZS.exists() else new_zs
        out_zs.to_parquet(OUT_ZS, index=False)

    print("[image] wrote", OUT_EMB)
    print("[image] wrote", OUT_ZS)


if __name__ == "__main__":
    main()
