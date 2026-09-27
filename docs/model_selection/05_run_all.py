"""Step 5: run 04_bench.py for every candidate (one fresh process each, in sequence)
and merge everything into results.json.

Run:  python 05_run_all.py            (all candidates)
      python 05_run_all.py <repo_id>  (only the named ones, then re-merge)
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

PLAN = [
    ("seqcls_text", "ealvaradob/bert-finetuned-phishing"),
    ("seqcls_text", "cybersectony/phishing-email-detection-distilbert_v2.4.1"),
    ("seqcls_text", "aamoshdahal/email-phishing-distilbert-finetuned"),
    ("seqcls_text", "ElSlay/BERT-Phishing-Email-Model"),
    ("seqcls_url", "CrabInHoney/urlbert-tiny-v4-phishing-classifier"),
    ("encoder_url", "amahdaouy/DomURLs_BERT"),
    ("seqcls_url", "ealvaradob/bert-phishing-url"),
    ("onnx_url", "pirocheto/phishing-url-detection"),
    ("seqcls_url", "kmack/malicious-url-detection"),
    ("clip", "openai/clip-vit-base-patch32"),
]


def load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    meta = load("metadata.json")
    only = sys.argv[1:]
    for kind, repo in PLAN:
        if only and repo not in only:
            continue
        rev = meta[repo]["revision"]
        t0 = time.perf_counter()
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, "04_bench.py"), kind, repo, rev],
            capture_output=True, text=True, cwd=HERE,
        )
        tail = (r.stdout.strip().splitlines() or [""])[-1]
        print(f"[{time.perf_counter() - t0:6.1f}s] rc={r.returncode} {tail}", flush=True)
        if r.returncode != 0:
            err = {"repo_id": repo, "revision": rev, "kind": kind, "status": "FAILED",
                   "error": "process exited with code %d" % r.returncode,
                   "stderr_tail": r.stderr.strip().splitlines()[-8:]}
            os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
            with open(os.path.join(HERE, "results", repo.replace("/", "__") + ".json"), "w", encoding="utf-8") as f:
                json.dump(err, f, indent=2)

    downloads = load("downloads.json")
    first = load("downloads_first_attempt.json")
    merged = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "candidates": []}
    for kind, repo in PLAN:
        m = meta.get(repo, {})
        rec = {
            "repo_id": repo,
            "branch": m.get("branch"),
            "revision": m.get("revision"),
            "licence_from_card_metadata": m.get("licence"),
            "gated": m.get("gated"),
            "card_datasets": m.get("card_datasets"),
            "card_base_model": m.get("card_base_model"),
            "repo_total_bytes_all_files": m.get("repo_total_bytes"),
            "download": downloads.get(repo),
            "download_first_attempt": first.get(repo),
            "bench": load(os.path.join("results", repo.replace("/", "__") + ".json")),
        }
        merged["candidates"].append(rec)
    merged["extra_checks"] = load("extra_checks.json")  # from 06_extra_checks.py, if it was run
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print("wrote results.json")


if __name__ == "__main__":
    main()
