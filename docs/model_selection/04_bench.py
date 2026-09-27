"""Step 4: load one candidate, run the harmless inputs, measure timings.

One candidate per process so that load time, cold inference and peak memory
are clean. Use 05_run_all.py to run every candidate.

Usage:
  python 04_bench.py <kind> <repo_id> <revision>
kinds:
  seqcls_text   sequence classifier, fed MESSAGES (and URLS as a side check)
  seqcls_url    sequence classifier, fed URLS (raw and scheme-stripped)
  encoder_url   bare encoder, checks for a classification head, extracts embeddings
  onnx_url      scikit-learn model exported to ONNX, fed URLS
  clip          CLIP zero-shot and image embeddings on images/*.png

Everything is offline after download (HF_HUB_OFFLINE=1), CPU only.

Timing notes:
  import_ms is the time to import torch, transformers and the model classes.
  load_ms is from_pretrained only (imports already paid). Weights are memory
  mapped, so part of the real cost shows up in cold_first_inference_ms.
  load_ms.model_reload_same_process is a second load in the same process.
  All runs were made with the files already in the operating system file
  cache (they had just been downloaded), so a load after a reboot will be slower.
Writes results/<repo>.json
"""
import json
import os
import statistics
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import psutil  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spike_inputs import LONG_TEXT, MESSAGES, URLS, strip_scheme  # noqa: E402

PROC = psutil.Process()
WARM_RUNS = 30


def mem_mb():
    mi = PROC.memory_info()
    return {
        "rss_mb": round(mi.rss / 1e6, 1),
        "peak_working_set_mb": round(getattr(mi, "peak_wset", mi.rss) / 1e6, 1),
    }


def timed(fn):
    t0 = time.perf_counter()
    r = fn()
    return r, (time.perf_counter() - t0) * 1000.0


def stats(ms):
    s = sorted(ms)
    return {
        "runs": len(ms),
        "median_ms": round(statistics.median(ms), 2),
        "mean_ms": round(statistics.fmean(ms), 2),
        "min_ms": round(s[0], 2),
        "p95_ms": round(s[max(0, int(round(0.95 * len(s))) - 1)], 2),
        "max_ms": round(s[-1], 2),
    }


def warm(fn, items, runs=WARM_RUNS):
    out = []
    for i in range(runs):
        _, ms = timed(lambda: fn(items[i % len(items)]))
        out.append(ms)
    return stats(out)


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------------------
def bench_seqcls(repo, rev, kind, res):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    res["torch_threads"] = torch.get_num_threads()
    (tok), ms_tok = timed(lambda: AutoTokenizer.from_pretrained(repo, revision=rev))
    model, ms_model = timed(
        lambda: AutoModelForSequenceClassification.from_pretrained(repo, revision=rev)
    )
    model.eval()
    res["load_ms"] = {"tokenizer": round(ms_tok, 1), "model": round(ms_model, 1),
                      "total": round(ms_tok + ms_model, 1)}
    res["tokenizer_class"] = type(tok).__name__
    res["model_class"] = type(model).__name__
    res["params"] = n_params(model)
    res["id2label"] = {str(k): v for k, v in model.config.id2label.items()}
    res["problem_type"] = model.config.problem_type
    res["max_position_embeddings"] = getattr(model.config, "max_position_embeddings", None)
    res["tokenizer_model_max_length"] = int(min(tok.model_max_length, 10**9))
    max_len = int(min(tok.model_max_length, res["max_position_embeddings"] or 512, 512))
    res["max_length_used"] = max_len
    res["memory_after_load"] = mem_mb()

    def predict(text):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
        with torch.no_grad():
            logits = model(**enc).logits[0]
        return logits, int(enc["input_ids"].shape[1])

    def describe(text, expected):
        logits, ntok = predict(text)
        probs = torch.softmax(logits, dim=-1).tolist()
        top = int(max(range(len(probs)), key=lambda i: probs[i]))
        return {
            "input": text,
            "expected_by_hand": expected,
            "tokens": ntok,
            "logits": [round(float(x), 4) for x in logits.tolist()],
            "softmax": [round(float(p), 4) for p in probs],
            "top_id": top,
            "top_label": model.config.id2label[top],
        }

    if kind == "seqcls_text":
        primary, side = MESSAGES, URLS
    else:
        primary, side = URLS, []

    _, cold = timed(lambda: predict(primary[0][0]))
    res["cold_first_inference_ms"] = round(cold, 1)
    res["warm_short_inputs"] = warm(lambda t: predict(t), [p[0] for p in primary])
    res["samples"] = [describe(t, e) for t, e in primary]
    res["unk_check"] = [
        {"input": t, "tokens": tok.tokenize(t)[:40],
         "unk_count": tok.tokenize(t).count(tok.unk_token)}
        for t, _ in primary[:2] + primary[-2:]
    ]

    if kind == "seqcls_url":
        res["samples_scheme_stripped"] = [describe(strip_scheme(t), e) for t, e in URLS]
    else:
        res["side_check_urls"] = [describe(t, e) for t, e in side]
        _, ntok = predict(LONG_TEXT)
        res["long_input_tokens"] = ntok
        res["warm_long_input_512_tokens"] = warm(lambda t: predict(t), [LONG_TEXT], runs=20)
    res["memory_end"] = mem_mb()
    _, ms_re = timed(lambda: AutoModelForSequenceClassification.from_pretrained(repo, revision=rev))
    res["load_ms"]["model_reload_same_process"] = round(ms_re, 1)
    res["status"] = "WORKS"


# --------------------------------------------------------------------------
def bench_encoder(repo, rev, res):
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    res["torch_threads"] = torch.get_num_threads()
    cfg = AutoConfig.from_pretrained(repo, revision=rev)
    res["config_architectures"] = cfg.architectures
    res["config_id2label"] = {str(k): v for k, v in (cfg.id2label or {}).items()}
    tok, ms_tok = timed(lambda: AutoTokenizer.from_pretrained(repo, revision=rev))
    model, ms_model = timed(lambda: AutoModel.from_pretrained(repo, revision=rev))
    model.eval()
    res["load_ms"] = {"tokenizer": round(ms_tok, 1), "model": round(ms_model, 1),
                      "total": round(ms_tok + ms_model, 1)}
    res["tokenizer_class"] = type(tok).__name__
    res["model_class"] = type(model).__name__
    res["params"] = n_params(model)
    res["hidden_size"] = cfg.hidden_size
    res["memory_after_load"] = mem_mb()

    # Does the checkpoint contain a trained classification head?
    _, info = AutoModelForSequenceClassification.from_pretrained(
        repo, revision=rev, output_loading_info=True
    )
    res["head_check"] = {
        "missing_keys_when_loaded_as_classifier": info["missing_keys"],
        "unexpected_keys": info["unexpected_keys"][:20],
        "has_trained_classification_head": not any(
            k.startswith("classifier") for k in info["missing_keys"]
        ),
    }
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo, "model.safetensors", revision=rev)
    with safe_open(p, framework="pt") as f:
        keys = list(f.keys())
    res["head_check"]["checkpoint_top_level_prefixes"] = sorted({k.split(".")[0] for k in keys})
    res["head_check"]["checkpoint_non_encoder_keys"] = [
        k for k in keys if not k.startswith(("bert.embeddings", "bert.encoder", "embeddings", "encoder"))
    ]

    max_len = 128

    def embed(text):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
        with torch.no_grad():
            out = model(**enc).last_hidden_state[0]
        cls = out[0]
        mean = out.mean(dim=0)
        return cls.numpy(), mean.numpy(), int(enc["input_ids"].shape[1])

    urls = [u for u, _ in URLS]
    _, cold = timed(lambda: embed(urls[0]))
    res["cold_first_inference_ms"] = round(cold, 1)
    res["warm_short_inputs"] = warm(lambda t: embed(t), urls)
    embs = []
    samples = []
    for u, e in URLS:
        cls, mean, ntok = embed(u)
        embs.append(mean)
        samples.append({
            "input": u, "expected_by_hand": e, "tokens": ntok,
            "tokenized": tok.tokenize(u)[:40],
            "cls_norm": round(float(np.linalg.norm(cls)), 4),
            "mean_pool_norm": round(float(np.linalg.norm(mean)), 4),
            "mean_pool_first5": [round(float(x), 4) for x in mean[:5]],
        })
    res["samples"] = samples
    X = np.stack(embs)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    res["embedding_shape"] = list(X.shape)
    res["cosine_similarity_matrix_mean_pool"] = [[round(float(v), 4) for v in row] for row in (Xn @ Xn.T)]

    # Mechanical check only: can scikit-learn fit a logistic head on these vectors?
    from sklearn.linear_model import LogisticRegression
    y = [0 if e == "benign" else 1 for _, e in URLS]
    clf, ms_fit = timed(lambda: LogisticRegression(max_iter=1000).fit(X, y))
    res["logistic_head_mechanical_check"] = {
        "note": "Fitted on the same 6 toy vectors it is scored on. Proves the plumbing only. Not an accuracy figure.",
        "fit_ms": round(ms_fit, 1),
        "coef_shape": list(clf.coef_.shape),
    }
    res["memory_end"] = mem_mb()
    _, ms_re = timed(lambda: AutoModel.from_pretrained(repo, revision=rev))
    res["load_ms"]["model_reload_same_process"] = round(ms_re, 1)
    res["status"] = "NEEDS HEAD" if not res["head_check"]["has_trained_classification_head"] else "WORKS"


# --------------------------------------------------------------------------
def bench_onnx(repo, rev, res):
    import numpy as np
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    res["onnxruntime_version"] = ort.__version__
    path = hf_hub_download(repo, "model.onnx", revision=rev)
    sess, ms_load = timed(lambda: ort.InferenceSession(path, providers=["CPUExecutionProvider"]))
    res["load_ms"] = {"total": round(ms_load, 1)}
    res["onnx_inputs"] = [(i.name, i.type, str(i.shape)) for i in sess.get_inputs()]
    res["onnx_outputs"] = [(o.name, o.type, str(o.shape)) for o in sess.get_outputs()]
    res["params"] = None
    res["params_note"] = "Not a neural network. TF-IDF features plus linear SVM exported to ONNX, parameter count not computed."
    res["id2label"] = {"0": "legitimate (column 0 of probabilities)", "1": "phishing (column 1 of probabilities)"}
    res["memory_after_load"] = mem_mb()

    def predict(u):
        return sess.run(None, {"inputs": np.array([u], dtype="str")})

    urls = [u for u, _ in URLS]
    _, cold = timed(lambda: predict(urls[0]))
    res["cold_first_inference_ms"] = round(cold, 2)
    res["warm_short_inputs"] = warm(lambda t: predict(t), urls)

    def describe(u, e):
        out = predict(u)
        label = out[0]
        proba = out[1]
        return {"input": u, "expected_by_hand": e,
                "raw_label_output": str(label.tolist() if hasattr(label, "tolist") else label),
                "probabilities": [round(float(x), 4) for x in np.asarray(proba)[0].tolist()]}

    res["samples"] = [describe(u, e) for u, e in URLS]
    res["samples_scheme_stripped"] = [describe(strip_scheme(u), e) for u, e in URLS]
    res["memory_end"] = mem_mb()
    res["status"] = "WORKS"


# --------------------------------------------------------------------------
ZS_PAIR = ["a login page asking for a password", "an ordinary web page"]
ZS_SET = [
    "a screenshot of a login page asking for a password",
    "a screenshot of a web page asking for credit card details",
    "a screenshot of a text message asking to confirm card details",
    "a screenshot of a news article or blog post",
    "a screenshot of a table of information",
]


def bench_clip(repo, rev, res):
    import glob

    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    res["torch_threads"] = torch.get_num_threads()
    proc, ms_proc = timed(lambda: CLIPProcessor.from_pretrained(repo, revision=rev))
    model, ms_model = timed(lambda: CLIPModel.from_pretrained(repo, revision=rev))
    model.eval()
    res["load_ms"] = {"processor": round(ms_proc, 1), "model": round(ms_model, 1),
                      "total": round(ms_proc + ms_model, 1)}
    res["image_processor_class"] = type(proc.image_processor).__name__
    res["params"] = n_params(model)
    res["params_vision_tower"] = n_params(model.vision_model) + n_params(model.visual_projection)
    res["params_text_tower"] = n_params(model.text_model) + n_params(model.text_projection)
    res["id2label"] = None
    res["projection_dim"] = model.config.projection_dim
    res["memory_after_load"] = mem_mb()

    paths = sorted(glob.glob(os.path.join(HERE, "images", "*.png")))
    images = {os.path.basename(p): Image.open(p).convert("RGB") for p in paths}
    names = list(images)

    def text_feats(prompts):
        enc = proc(text=prompts, return_tensors="pt", padding=True)
        with torch.no_grad():
            t = model.get_text_features(**enc)
        return t / t.norm(dim=-1, keepdim=True)

    def image_feats(img):
        enc = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            v = model.get_image_features(**enc)
        return v / v.norm(dim=-1, keepdim=True)

    def zero_shot_full(img, prompts):
        enc = proc(text=prompts, images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(**enc)
        return out.logits_per_image[0].softmax(dim=-1)

    # (a) zero-shot
    _, cold = timed(lambda: zero_shot_full(images[names[0]], ZS_PAIR))
    res["cold_first_inference_ms"] = round(cold, 1)
    res["zero_shot"] = {
        "warm_full_text_plus_image": warm(lambda n: zero_shot_full(images[n], ZS_PAIR), names),
        "prompts_pair": ZS_PAIR,
        "prompts_set": ZS_SET,
        "samples_pair": [],
        "samples_set": [],
    }
    scale = model.logit_scale.exp().item()
    res["zero_shot"]["logit_scale"] = round(scale, 4)
    tp, ms_tp = timed(lambda: text_feats(ZS_PAIR))
    ts = text_feats(ZS_SET)
    res["zero_shot"]["text_prompt_encoding_ms_pair_once"] = round(ms_tp, 1)
    for n in names:
        v = image_feats(images[n])
        cos_p = (v @ tp.T)[0]
        cos_s = (v @ ts.T)[0]
        res["zero_shot"]["samples_pair"].append({
            "image": "images/" + n,
            "cosine": [round(float(x), 4) for x in cos_p.tolist()],
            "softmax": [round(float(x), 4) for x in (scale * cos_p).softmax(-1).tolist()],
        })
        sm = (scale * cos_s).softmax(-1).tolist()
        res["zero_shot"]["samples_set"].append({
            "image": "images/" + n,
            "cosine": [round(float(x), 4) for x in cos_s.tolist()],
            "softmax": [round(float(x), 4) for x in sm],
            "top_prompt": ZS_SET[int(np.argmax(sm))],
        })

    # (b) image embeddings for a later logistic head
    res["image_embeddings"] = {
        "warm_image_only": warm(lambda n: image_feats(images[n]), names),
        "samples": [],
    }
    E = []
    for n in names:
        v = image_feats(images[n])[0].numpy()
        E.append(v)
        res["image_embeddings"]["samples"].append({
            "image": "images/" + n, "dim": int(v.shape[0]), "dtype": str(v.dtype),
            "l2_norm_after_normalisation": round(float(np.linalg.norm(v)), 4),
            "first5": [round(float(x), 4) for x in v[:5]],
        })
    E = np.stack(E)
    res["image_embeddings"]["order"] = names
    res["image_embeddings"]["cosine_similarity_matrix"] = [[round(float(x), 4) for x in r] for r in (E @ E.T)]
    res["warm_short_inputs"] = res["image_embeddings"]["warm_image_only"]
    res["memory_end"] = mem_mb()
    _, ms_re = timed(lambda: CLIPModel.from_pretrained(repo, revision=rev))
    res["load_ms"]["model_reload_same_process"] = round(ms_re, 1)
    res["status"] = "WORKS"


# --------------------------------------------------------------------------
def main():
    kind, repo, rev = sys.argv[1], sys.argv[2], sys.argv[3]
    import platform

    # Imports are timed separately from model loading. transformers imports its
    # model code lazily, so the concrete classes are touched here on purpose.
    t0 = time.perf_counter()
    import torch
    t1 = time.perf_counter()
    import transformers
    if kind != "onnx_url":
        from transformers import (  # noqa: F401
            AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer,
            BertForSequenceClassification, BertModel, BertTokenizerFast,
            CLIPModel, CLIPProcessor, DistilBertForSequenceClassification,
            DistilBertTokenizerFast, PreTrainedTokenizerFast,
        )
    t2 = time.perf_counter()
    import_ms = {"torch": round((t1 - t0) * 1000, 1),
                 "transformers_and_model_classes": round((t2 - t1) * 1000, 1),
                 "total": round((t2 - t0) * 1000, 1),
                 "USE_TORCH_env": os.environ.get("USE_TORCH")}

    res = {
        "repo_id": repo, "revision": rev, "kind": kind,
        "env": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "device": "cpu",
            "logical_cpus": os.cpu_count(), "offline_mode": os.environ.get("HF_HUB_OFFLINE"),
        },
        "import_ms": import_ms,
        "memory_before_load": mem_mb(),
    }
    import warnings
    caught = []
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            if kind in ("seqcls_text", "seqcls_url"):
                bench_seqcls(repo, rev, kind, res)
            elif kind == "encoder_url":
                bench_encoder(repo, rev, res)
            elif kind == "onnx_url":
                bench_onnx(repo, rev, res)
            elif kind == "clip":
                bench_clip(repo, rev, res)
            else:
                raise SystemExit("unknown kind " + kind)
            caught = sorted({str(x.message)[:300] for x in w})
    except Exception as e:  # noqa: BLE001
        import traceback
        res["status"] = "FAILED"
        res["error"] = repr(e)
        res["traceback_tail"] = traceback.format_exc().splitlines()[-6:]
    res["python_warnings"] = caught
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", repo.replace("/", "__") + ".json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(repo, res["status"], res.get("load_ms"), res.get("warm_short_inputs"))


if __name__ == "__main__":
    main()
