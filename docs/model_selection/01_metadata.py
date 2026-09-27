"""Step 1: collect hub metadata for every candidate (no model weights downloaded here).

Writes metadata.json and saves each model card to cards/<repo>.md so the
"trained on" claims can be read and quoted.

Run:  python 01_metadata.py
"""
import json
import os
import sys

from huggingface_hub import HfApi, hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
CARDS = os.path.join(HERE, "cards")
os.makedirs(CARDS, exist_ok=True)

CANDIDATES = {
    "text": [
        "ealvaradob/bert-finetuned-phishing",
        "cybersectony/phishing-email-detection-distilbert_v2.4.1",
        "aamoshdahal/email-phishing-distilbert-finetuned",
        "ElSlay/BERT-Phishing-Email-Model",
    ],
    "url": [
        "CrabInHoney/urlbert-tiny-v4-phishing-classifier",
        "amahdaouy/DomURLs_BERT",
        "ealvaradob/bert-phishing-url",
        "pirocheto/phishing-url-detection",
        "kmack/malicious-url-detection",
    ],
    "image": [
        "openai/clip-vit-base-patch32",
    ],
}


def main():
    api = HfApi()
    out = {}
    for branch, repos in CANDIDATES.items():
        for repo in repos:
            rec = {"branch": branch, "repo_id": repo}
            try:
                info = api.model_info(repo, files_metadata=True)
                rec["revision"] = info.sha
                rec["gated"] = info.gated
                rec["private"] = info.private
                rec["pipeline_tag"] = info.pipeline_tag
                rec["library_name"] = info.library_name
                rec["downloads"] = info.downloads
                rec["last_modified"] = str(info.last_modified)
                card = info.card_data.to_dict() if info.card_data else {}
                rec["licence"] = card.get("license")
                rec["card_datasets"] = card.get("datasets")
                rec["card_base_model"] = card.get("base_model")
                rec["tags"] = info.tags
                rec["files"] = [
                    {"name": s.rfilename, "bytes": s.size} for s in info.siblings
                ]
                rec["repo_total_bytes"] = sum(s.size or 0 for s in info.siblings)
                try:
                    p = hf_hub_download(repo, "README.md", revision=info.sha)
                    with open(p, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read()
                    with open(
                        os.path.join(CARDS, repo.replace("/", "__") + ".md"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write(text)
                    rec["card_chars"] = len(text)
                except Exception as e:  # noqa: BLE001
                    rec["card_error"] = repr(e)
                rec["metadata_status"] = "OK"
            except Exception as e:  # noqa: BLE001
                rec["metadata_status"] = "ERROR"
                rec["metadata_error"] = repr(e)
            out[repo] = rec
            print(
                repo,
                rec.get("metadata_status"),
                rec.get("revision"),
                rec.get("licence"),
                rec.get("repo_total_bytes"),
                file=sys.stderr,
            )
    with open(os.path.join(HERE, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
