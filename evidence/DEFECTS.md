# Defects log

## Application defects

### D1. Outcome badge shown without its colour

- **Symptom:** the outcome badge in the result panel showed the outcome text
  but no background colour, for every outcome. The outcome was therefore
  communicated by text only, not by text and colour as designed.
- **How it was found:** by inspecting the captured interface screenshots
  against the design requirement during a review of the implementation. No
  automated test covered the link between outcome labels and style rules.
- **Reproduce:** run any analysis in the earlier version and inspect the
  badge element. Its class is `outcome-Limited.indicators` (one class name
  containing a dot).
- **Cause:** `outcomeClass()` in `static/app.js` replaced the space in the
  outcome label with a dot. The stylesheet rule `.outcome-Limited.indicators`
  is read by the browser as two classes, `outcome-Limited` and `indicators`,
  so it never matched the single class on the element.
- **Fix:** the class is now the lower-case label with spaces replaced by
  hyphens (for example `outcome-limited-indicators`), and the four stylesheet
  rules were renamed to match.
- **Regression test:** `tests/unit/test_outcome_styles.py` checks that every
  outcome label defined in `schemas.py` maps to a valid single class name
  with a matching rule in `style.css`, and that `app.js` uses the same rule.

### D2. Fresh installation failed seven API tests

- **Symptom:** in a clean virtual environment installed only from
  `requirements.txt`, 69 of 76 unit tests passed and 7 stopped with an
  error in `tests/unit/test_api.py`:
  `TypeError: Client.__init__() got an unexpected keyword argument 'app'`.
- **How it was found:** by a clean-environment installation test, carried out
  to check that the project installs on a machine other than the development
  machine. The full record is in `evidence/CLEAN_INSTALL.md`.
- **Cause:** `requirements.txt` did not pin `httpx`. A fresh install resolved
  a newer `httpx` release that no longer accepts the `app` argument used by
  the test client of the pinned FastAPI version. The development environment
  had hidden the problem because it could see packages installed outside the
  project.
- **Fix:** `httpx==0.27.2` is pinned in `requirements.txt`, and the CPU build
  of PyTorch is resolved through an explicit package index line.
- **Regression evidence:** the clean-environment test was repeated after the
  fix: 76 passed, 0 failed, and one real analysis through the API completed
  with all three models.

The adapters, orchestrator, fusion, policy and API worked against the
fixtures on the first real-model run.

## Test suite issues

Two issues were found and fixed in the test suite itself while writing it
(not application defects, listed here for a complete record):

1. **Symptom:** `test_url_model_truncation_flag_on_long_url` failed,
   expecting a 500-character URL of a repeated single letter to exceed the
   64-token limit; it tokenized to only 17 tokens.
   **Cause:** the URL model's WordPiece-style tokenizer merges long runs of
   the same character into very few tokens, so a repeated character is not a
   realistic way to force truncation.
   **Fix:** the test now builds a 400-character path from a mixed alphabet,
   punctuation and digits, which reliably produces over 64 tokens
   (`tests/integration/test_real_models.py`).

2. **Symptom:** two unit tests were logically wrong when first written
   (`test_case_blank_strings_count_as_not_provided` called `validate_case`
   in a way that raised outside the `pytest.raises` block; the network
   monkeypatch test in `test_api.py` failed because Windows' asyncio
   `ProactorEventLoop` opens a loopback socket pair for its own plumbing,
   which is unrelated to whether the application code makes a network call).
   **Fix:** corrected the first test's structure, and moved the
   authoritative no-network proof to `tests/unit/test_no_network.py`, which
   calls the orchestrator directly instead of through the async test client,
   avoiding the unrelated event-loop socket use while still proving that
   validation, the URL feature extractor and the orchestrator never call
   `socket.socket.connect` or `socket.getaddrinfo`.

Environment notes (not defects, recorded for reproducibility):
- `python-multipart` and `httpx` were required by FastAPI's `UploadFile`
  form handling and test client respectively; both were already present on
  this machine's Python installation and were confirmed importable in the
  project venv.
- The project was installed into the venv in editable mode
  (`pip install -e . --no-deps`) so that `import phishlens` resolves from
  `src/phishlens` without modifying `sys.path` by hand in every entry point.
