"""Step 6: small follow-up checks. Writes extra_checks.json.

  a. import time of transformers with and without USE_TORCH=1 / USE_TF=0 / USE_FLAX=0
  b. whether a safetensors conversion exists on a pull request ref for the
     two repos that only ship pytorch_model.bin
  c. how transformers 4.48 calls torch.load (weights_only)
  d. dataset cards of the training sets named by the model cards (saved to cards/)

Run:  python 06_extra_checks.py
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}

CODE = (
    "import time;t=time.perf_counter();import torch;"
    "from transformers import AutoTokenizer, BertForSequenceClassification, CLIPModel, CLIPProcessor;"
    "import sys;print(round((time.perf_counter()-t)*1000,1), sorted(m for m in ('tensorflow','jax','flax') if m in sys.modules))"
)
out["import_time_ms"] = {}
for name, extra in (("default_env", {}), ("USE_TORCH=1 USE_TF=0 USE_FLAX=0", {"USE_TORCH": "1", "USE_TF": "0", "USE_FLAX": "0"})):
    runs = []
    for _ in range(3):
        env = dict(os.environ, **extra)
        r = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, env=env)
        runs.append(r.stdout.strip() or r.stderr.strip()[-200:])
    out["import_time_ms"][name] = runs

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

api = HfApi()
out["pull_request_refs"] = {}
for repo in ("ealvaradob/bert-finetuned-phishing", "openai/clip-vit-base-patch32"):
    try:
        refs = api.list_repo_refs(repo, include_pull_requests=True)
        found = []
        for pr in refs.pull_requests or []:
            try:
                files = api.list_repo_files(repo, revision=pr.ref)
                if any(f.endswith(".safetensors") for f in files):
                    found.append({"ref": pr.ref, "commit": pr.target_commit})
            except Exception as e:  # noqa: BLE001
                found.append({"ref": pr.ref, "error": repr(e)[:120]})
        out["pull_request_refs"][repo] = {"pull_requests": len(refs.pull_requests or []), "with_safetensors": found}
    except Exception as e:  # noqa: BLE001
        out["pull_request_refs"][repo] = {"error": repr(e)[:200]}

import transformers  # noqa: E402

src = open(os.path.join(os.path.dirname(transformers.__file__), "modeling_utils.py"), encoding="utf-8").read()
out["transformers_torch_load"] = {
    "weights_only_mentions": len(re.findall(r"weights_only", src)),
    "lines": [ln.strip() for ln in src.splitlines() if "weights_only" in ln][:12],
}

out["dataset_cards"] = {}
for ds in ("ealvaradob/phishing-dataset", "cybersectony/PhishingEmailDetectionv2.0",
           "zefang-liu/phishing-email-dataset", "pirocheto/phishing-url", "kmack/Phishing_urls"):
    try:
        info = api.dataset_info(ds)
        p = hf_hub_download(ds, "README.md", repo_type="dataset", revision=info.sha)
        text = open(p, encoding="utf-8", errors="replace").read()
        with open(os.path.join(HERE, "cards", "dataset__" + ds.replace("/", "__") + ".md"), "w", encoding="utf-8") as f:
            f.write(text)
        card = info.card_data.to_dict() if info.card_data else {}
        out["dataset_cards"][ds] = {"revision": info.sha, "licence": card.get("license"), "gated": info.gated,
                                    "files": [s.rfilename for s in info.siblings][:30], "card_chars": len(text)}
    except Exception as e:  # noqa: BLE001
        out["dataset_cards"][ds] = {"error": repr(e)[:200]}

with open(os.path.join(HERE, "extra_checks.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
