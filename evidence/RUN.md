# Evidence run log

Date: 2026-09-27 (timing and screenshots re-measured on an idle machine)
Platform: Windows, Python 3.11.3, CPU only, torch 2.2.2+cpu, transformers 4.48.0.
Venv: `.venv` (created with --system-site-packages, reused; the
package itself installed into it with `pip install -e . --no-deps`).

This re-run followed installing the fitted fusion and image-head parameters
into `models/fusion.json` and `models/image_head.json` (see
`evidence/fusion_consistency.txt` for the consistency check against the
evaluation's fused scores). The unfitted default that used to ship as
`models/fusion.json` is preserved at `models/fusion.default.json`.

## Fusion consistency check (app fusion/policy vs. the evaluation)

Command:
```
.venv\Scripts\python.exe scripts\check_fusion_consistency.py
```
Output saved to `evidence/fusion_consistency.txt`. Result: 20 locked test
cases checked, maximum absolute score difference 0.000e+00, all three-outcome
labels matched. No defect found in the fusion/policy path itself.

## Unit tests (fake adapters, no models loaded, no network)

Command:
```
.venv\Scripts\python.exe -m pytest tests/unit -q --junitxml=evidence/unit_junit.xml
```
Output saved to `evidence/unit_output.txt`. Result: 85 passed, 1 warning
(a Starlette deprecation notice about httpx in TestClient, not a failure),
confirmed again on 2026-09-27.

## Integration tests (marker `slow`, three real pretrained models)

Command:
```
.venv\Scripts\python.exe -m pytest tests/integration -m slow -q --junitxml=evidence/integration_junit.xml
```
Output saved to `evidence/integration_output.txt`. Result: 7 passed,
confirmed again on 2026-09-27.
These load `ealvaradob/bert-finetuned-phishing`,
`CrabInHoney/urlbert-tiny-v4-phishing-classifier` and
`openai/clip-vit-base-patch32` from the local Hugging Face cache
(`HF_HUB_OFFLINE=1`) and run real inference against `fixtures/cases.json`
and `fixtures/images/*.png`.

## Manual end-to-end timing (real models, one process)

Command:
```
.venv\Scripts\python.exe scripts\e2e_timing.py
```
Output saved to `evidence/e2e_timing.txt`. Loads all three real models once,
then analyzes one fixture case (url + message text + screenshot together)
in the same process and prints per-branch and total latency.

Measured 2026-09-27 on an idle machine (no other CPU-heavy process
running): model load 4578.5 ms; one full case analysis (url + text +
image, models already warm) 427.89 ms; combined 5006.4 ms.

This single-case run is a smoke check, not the reference timing. The
reference values come from `evaluation/outputs/tables/runtime.csv` and
`runtime_branches.csv` (200 cases each, also measured idle): end-to-end
warm median 1,165.6 ms, warm p95 2,708.7 ms, cold first request 7,721.8 ms;
per-branch warm medians 7.6 ms URL, 1,046.1 ms text (on long page text,
not the short fixture messages used above), 103.6 ms image, model load
4,335.3 ms.

## Screenshots (Playwright, real server, real models)

Command:
```
set PHISHLENS_SCREENSHOT_PORT=8011
.venv\Scripts\python.exe scripts\capture_screenshots.py
```
Chromium was already installed for this account's Playwright cache, so no
network install was needed. Re-captured 2026-09-27 on an idle machine so
the latency column shown in each screenshot reflects idle-machine timing.
6 screenshots were captured to `evidence/screenshots/` with a
`screenshots.json` sidecar: an empty form (`01_empty_form.png`), a
complete three-input harmless case (`02_complete_harmless_case.png`,
outcome "Limited indicators"), a partial url-only case
(`03_partial_case_url_only.png`), a validation error
(`04_validation_error.png`), a "Review needed" case that arose naturally
from a fixture combining a suspicious-shaped URL with an ordinary message
(`05_review_needed_case.png`), and a "High concern" case combining the
suspicious URL, an urgent message, and the fixture login-page screenshot
(`06_high_concern_case.png`). All three outcomes match the prior capture;
only the branch latency values changed.

## Clean installation test

See `evidence/CLEAN_INSTALL.md` for the from-scratch install (plain venv,
no `--system-site-packages`, installed from `requirements.txt` alone) that
proves the project runs end to end without relying on globally installed
packages.
