[README.md](https://github.com/user-attachments/files/32706335/README.md)
# PhishLens

PhishLens is a local phishing triage web application. It takes an optional
URL, message text and screenshot, runs each through a small pretrained
model, fuses the three scores, and returns one of four outcomes with a
plain-language explanation built only from the evidence actually produced.

It is a triage aid, not a verdict, and it never says an item is safe or
legitimate.

## Hard rule: nothing submitted is ever fetched

The submitted URL is treated purely as a string. Nothing in this
application performs a DNS lookup, opens a TCP connection, or otherwise
contacts the URL. The server binds to `127.0.0.1` only. This is proven by
an automated test (`tests/unit/test_no_network.py`) that monkeypatches
`socket.socket.connect` and `socket.getaddrinfo` to raise, then runs
validation and a full orchestrator analysis; either function being called
would fail the test. Submitted content (url, text, screenshot bytes) is
processed in memory only and is never written to disk.

## Models

All three run locally, offline, on CPU, from pinned revisions in the local
Hugging Face cache. Licence terms are as stated on each model's Hugging
Face model card at the time of writing; check the card itself for the
current terms before any redistribution or commercial use.

| Branch | Model | Revision | Notes |
|---|---|---|---|
| URL | `CrabInHoney/urlbert-tiny-v4-phishing-classifier` | `fd962a5cd04e20ceed46aa8a6e31a2b40d4c8916` | Apache-2.0 (per model card). Sequence classifier over the URL string, max 64 tokens. The model card's label names are generic (`LABEL_0`/`LABEL_1`); the mapping used here (0 good, 1 phishing) is hard coded from the model card text. Always called with the full URL including its scheme. |
| Text | `ealvaradob/bert-finetuned-phishing` | `fa8fb73a007174c410ab7160d4e4c6e6b8d998d4` | Apache-2.0 (per model card). Sequence classifier over message text, max 512 tokens, labels 0 benign / 1 phishing. |
| Image | `openai/clip-vit-base-patch32` | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | MIT (per model card). Used for 512-dim image embeddings, scored either zero-shot against a fixed prompt set or (if `models/image_head.json` exists) through a fitted logistic head. |

## How a case is scored

1. Each provided input is validated (see below) and passed to its adapter.
   A branch that raises an exception is reported as `failed`; the other
   branches still run (failure isolation). A branch with no input is
   `unavailable`.
2. The usable branch probabilities are fused into one score. Method is
   selectable (`mean`, `max`, `best_single`, `learned`); the shipped
   `models/fusion.json` uses the fitted `learned` method (a regularised
   logistic regression over the six-feature vector, with Platt calibration:
   a=1.00625, b=-0.18000) and fitted thresholds (low 0.3519 / high 0.7633)
   and disagreement gap (0.9967). These were fitted on 694 development
   linked cases and evaluated once on a locked 299-case test set; see
   `evaluation/` for the full methodology, fitting scripts and results, and
   `evaluation/outputs/fitted/fusion.json` and `image_head.json` for the
   frozen parameter files these were copied from. The unfitted provisional
   default (`mean` fusion, thresholds 0.30 / 0.70, disagreement gap 0.50) is
   kept at `models/fusion.default.json` for reference and is no longer what
   ships. The image branch also uses a fitted logistic head
   (`models/image_head.json`) rather than zero-shot scoring.
3. Policy turns the fused score into one of four outcomes:
   - **High concern**: fused score at or above the high threshold.
   - **Limited indicators**: fused score at or below the low threshold.
     This does not mean the item is legitimate.
   - **Review needed**: fused score between the thresholds, or triggered by
     the disagreement gate (two or more usable branches whose scores differ
     by more than the disagreement gap, regardless of the fused score).
   - **Analysis unavailable**: no branch produced a usable score.
4. Explanation sentences are generated from a fixed set of templates, each
   filled only with values actually present in that case's evidence.

## Validation rules

- URL: must parse with scheme `http` or `https` and a host name, at most
  2,048 characters, no whitespace or control characters. The exact string
  submitted is what gets analyzed; nothing is normalised or rewritten.
- Message text: at most 10,000 characters.
- Screenshot: PNG or JPEG, at most 5 MB, opened and decoded with Pillow. A
  pixel-count and per-side cap guards against decompression bombs.
- At least one of the three must be provided.

## Install and run (Windows)

Requires Python 3.11 on PATH. CPU only; no GPU is required or used.

The easy way: run `setup.bat` once (creates `.venv`, installs
dependencies, builds fixtures, and downloads the models, about 2.3 GB), then
`run.bat` every time after that to start the server.

The manual equivalent, if you want to see each step:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\pip.exe install -r requirements.txt
.venv\Scripts\pip.exe install -e . --no-deps
.venv\Scripts\python.exe fixtures\make_fixtures.py
.venv\Scripts\python.exe scripts\download_models.py
```

`requirements.txt` adds `--extra-index-url https://download.pytorch.org/whl/cpu`
so `pip` resolves `torch==2.2.2` as the CPU-only wheel instead of a
multi-GB CUDA build. `httpx` is pinned to `0.27.2`; a newer `httpx` removes
a shortcut that `fastapi==0.100.1`'s test client depends on, which broke
the API tests in a from-scratch install (see `evidence/CLEAN_INSTALL.md`).

`scripts\download_models.py` downloads the three pinned model revisions
(about 2.3 GB total) into the local Hugging Face cache, retries once on
failure (Windows can raise a symlink-privilege `OSError` on the first
attempt and succeed on the second), and then reloads each model offline to
confirm it actually works. This only needs to be run once per machine.

Run the server:

```
.venv\Scripts\python.exe scripts\run_server.py --preload
```

or just `run.bat`. Then open `http://127.0.0.1:8000/` in a browser.
`--preload` loads all three models before the server starts accepting
requests; without it, each model loads on its first use.

This install path was proven on a real clean environment: a fresh
`.venv` with no access to globally installed packages, on the same
machine, installing only from `requirements.txt`. See
`evidence/CLEAN_INSTALL.md` for the exact commands, the one bug found and
fixed (the `httpx` pin above), and a real end-to-end API call against the
real models.

### API

- `GET /` the plain HTML/JS interface.
- `POST /api/analyze` multipart form fields `url`, `message_text`,
  `screenshot` (all optional, at least one required). Returns the case
  result as JSON.
- `GET /api/health` model load status, ids, revisions, and the active
  fusion configuration, including whether it is provisional.

## Tests

Unit tests use fake adapters (no torch, no transformers, no network) and
cover validation, the fusion feature vector order and each fusion method,
policy threshold boundaries and the disagreement gate, missing-modality and
failure-isolation behaviour in the orchestrator, explanation text, the API,
and the no-network guarantee:

```
.venv\Scripts\python.exe -m pytest tests/unit -q
```

Integration tests are marked `slow` and exercise the three real models
against the fixtures end to end:

```
.venv\Scripts\python.exe -m pytest tests/integration -m slow -q
```

Real pytest text output, JUnit XML, the exact commands and the date these
were run are saved under `evidence/`, along with a manual end-to-end timing
run (`evidence/e2e_timing.txt`) and Playwright screenshots of the running
app (`evidence/screenshots/`). See `evidence/RUN.md` for the full log and
`evidence/DEFECTS.md` for defects found while building.

## Structure

```
src/phishlens/
  config.py        model ids, revisions, paths, limits
  schemas.py        data structures shared across the app
  validation.py      input checks, no network access anywhere in this file
  adapters/           one module per branch (url, text, image) plus a shared base
  fusion.py           the four fusion methods and fusion.json read/write/fit
  policy.py            thresholds and the disagreement gate
  explanations.py       evidence-only explanation templates
  orchestrator.py       runs the three branches, fuses, decides, explains
  api.py                 FastAPI app: GET /, POST /api/analyze, GET /api/health
  static/                 plain HTML/CSS/JS interface, no build step
models/
  fusion.json              the active (shipped, fitted) fusion configuration
  fusion.default.json       the unfitted provisional default, kept for reference
  image_head.json           the fitted logistic head for the image branch
  manifest.json             model ids and revisions in one place
tests/
  unit/                     fake adapters, no models, no network
  integration/               marker "slow", the three real models
scripts/
  run_server.py               starts the app on 127.0.0.1
  download_models.py            downloads the three pinned models once, with retry
  capture_screenshots.py       Playwright screenshots of the real running app
  e2e_timing.py                 manual end-to-end latency measurement
fixtures/
  make_fixtures.py               builds cases.json and synthetic images
evidence/                       saved real test output, timing, screenshots, clean-install log
setup.bat                        one-time setup: venv, dependencies, fixtures, model download
run.bat                          starts the server
```

## Limitations

- The shipped `fusion.json` is fitted (`learned` method, Platt calibration,
  thresholds low 0.3519 / high 0.7633, disagreement gap 0.9967), fitted on
  694 development linked cases and evaluated once on a locked 299-case test
  set. See `evaluation/` for the fitting and evaluation pipeline (scripts,
  cached predictions, figures and tables) and `evaluation/outputs/fitted/`
  for the frozen source of the installed parameters. The earlier unfitted
  provisional default (`mean` fusion, thresholds 0.30 / 0.70, disagreement
  gap 0.50) is kept at `models/fusion.default.json` for reference; it is no
  longer what the app loads by default. To fall back to it, copy it over
  `models/fusion.json` (never edit the fitted file's values directly:
  re-fit with `fusion.fit()` against new labelled data instead).
- The image branch now uses the fitted logistic head at
  `models/image_head.json` rather than zero-shot prompt matching. Zero-shot
  scoring remains the automatic fallback if that file is ever removed.
- All three models are small, general-purpose phishing classifiers running
  on CPU with no ensembling beyond the fusion step. They will miss novel
  phishing patterns and can be wrong in both directions.
- The URL branch sees only the string; it cannot check the actual content
  of the destination, because the application never contacts it.
- Screenshot analysis works on a single static image and does not follow
  links, scroll, or otherwise render the page in any way.
- This is a local, single-user tool, not a production security control.
