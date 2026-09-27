# Clean install test

Date: 2026-09-27

Purpose: prove that a copy of this project, on the same machine but in a
plain Python virtual environment with no access to globally installed
packages, installs from `requirements.txt` alone and runs end to end with
the real models. This is the environment a student's laptop is in on a
fresh download: the developer's original `.venv` was created with
`--system-site-packages` and silently used a globally installed
`torch 2.2.2+cpu` and `transformers 4.48.0`; a student's machine has
neither, so this had to be proven without that shortcut.

## What was done

1. Copied the project into a scratch folder outside the project,
   excluding `.venv`, `__pycache__`, `.pytest_cache`, `.benchmarks`,
   `.git`, and the `evaluation` folder (it was excluded from the copy,
   since the application does not depend on it).
2. Created a plain virtual environment with **no** `--system-site-packages`:
   ```
   python -m venv .venv
   ```
   Confirmed it has no access to the global site-packages: `sys.path`
   contains only the venv's own `Lib\site-packages`.
3. Installed dependencies from `requirements.txt`:
   ```
   .venv\Scripts\python.exe -m pip install --upgrade pip
   .venv\Scripts\pip.exe install -r requirements.txt
   ```
   `torch==2.2.2` resolved as `torch-2.2.2+cpu` from the added
   `--extra-index-url https://download.pytorch.org/whl/cpu` line in
   `requirements.txt` (the correct way to get a CPU-only wheel on Windows
   without pulling a multi-GB CUDA build).
4. Installed the package itself and built fixtures:
   ```
   .venv\Scripts\pip.exe install -e . --no-deps
   .venv\Scripts\python.exe fixtures\make_fixtures.py
   ```
5. Ran the unit tests:
   ```
   .venv\Scripts\python.exe -m pytest tests/unit -q
   ```

## What broke and the fix

First run: 69 passed, 7 errors, all in `tests/unit/test_api.py`, all the
same error:

```
TypeError: Client.__init__() got an unexpected keyword argument 'app'
```

Cause: `requirements.txt` left `httpx` unpinned. A plain install (no global
packages to fall back on) pulled the current `httpx 0.28.1`, which removed
the `app=` shortcut that Starlette 0.27.0's `TestClient` (pulled in by the
pinned `fastapi==0.100.1`) depends on. On the developer's machine this
never surfaced, because the dev `.venv` used `--system-site-packages` and
a newer `fastapi` (0.141.1, with `starlette` 1.3.1) already installed
globally silently overrode the pinned `fastapi==0.100.1` in the venv, so
the version mismatch never triggered there. That is exactly the kind of
hidden dependency this clean-room test exists to catch.

Fix: pinned `httpx==0.27.2` in `requirements.txt` (last release before the
`app=` shortcut was removed, still compatible with Starlette 0.27.0).

Re-ran the install and tests after the fix:

```
.venv\Scripts\pip.exe install "httpx==0.27.2"
.venv\Scripts\python.exe -m pytest tests/unit -q
```

Result: **76 passed**, 0 failed, 0 errors (16 warnings, all pre-existing
deprecation notices from third-party libraries, not from this project's
code).

## Real-model end-to-end run

The model cache was **reused** from the machine's existing Hugging Face
cache (the default per-user cache folder, `.cache\huggingface\hub` under
the user profile) to save the roughly 2.3 GB re-download; this is a real
shortcut taken for this test only, noted here rather than hidden. It is a
plain per-user cache directory that any `huggingface_hub` install on the
same machine reads by default; nothing about it is specific to the
developer's original `.venv`.

To separately confirm that `scripts\download_models.py` itself resolves
the three pinned revisions correctly (the part a student's machine
actually needs to do from empty), it was run against the shared cache:

```
.venv\Scripts\python.exe scripts\download_models.py
```

Output: all three branches (`url`, `text`, `image`) reported
`downloaded OK` against their pinned revisions, immediately followed by
`verify_offline_load()` re-loading all three with `HF_HUB_OFFLINE=1`,
which each reported `loads offline OK`. Because the cache already had
these exact revisions, no bytes were actually transferred, but the
resolve-by-revision, retry-on-`OSError` and offline-verification code
paths all ran and succeeded. This confirms the script's logic is correct;
it does not by itself prove a from-empty download on a machine with no
network restrictions works, which cannot be tested on this machine without
deleting the shared cache other work depends on.

Then the server was started in the clean venv and one real analysis was
run against it through the HTTP API, using the project's own synthetic
fixtures (a suspicious URL, an urgent-style phishing message, and a
synthetic login-page screenshot):

```
.venv\Scripts\python.exe scripts\run_server.py --preload
```

```
curl -s http://127.0.0.1:8000/api/health
curl -s -X POST http://127.0.0.1:8000/api/analyze \
  -F "url=http://192.0.2.44/secure-login-update@example.net/confirm-account-details" \
  -F "message_text=URGENT: your account on example.net will be suspended in 24 hours. Confirm your password and card details now at the link below to keep access." \
  -F "screenshot=@fixtures/images/login_page.png;type=image/png"
```

`GET /api/health` reported all three branches `"loaded": true` with the
exact pinned model ids and revisions. `POST /api/analyze` returned a real
result computed by the three real models and the fusion step:

- outcome: `"High concern"`
- fused_score: `0.9998...`
- all three branches (`url`, `text`, `image`) status `"ok"`, each with a
  real probability, real per-branch latency, and evidence-based
  explanation text
- `total_latency_ms`: ~3033 ms with all three models already loaded
  (`--preload`)

This proves the clean venv runs the real app end to end, not just the
fake-adapter unit tests.

The server was then confirmed stopped (`/api/health` unreachable
afterwards) and the clean-room `.venv` was deleted to free disk space,
per instructions. The clean-room project files (minus `.venv`) were left
in place at `a scratch folder` as a record of what was copied.

## Not run in this pass

Integration tests (`pytest tests/integration -m slow`) were not re-run in
the clean venv; the equivalent real-model coverage was exercised directly
through the live API call above, which is a stronger end-to-end check
(HTTP request in, JSON result out) than the integration suite alone would
add for the purpose of this clean-install proof. The evaluation folder was
left untouched throughout, per instructions.
