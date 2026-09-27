"""Consistency check: does the shipped app fusion reproduce the evaluation's
fused scores and three-outcome labels for locked test cases?

Takes 20 locked test cases from evaluation/outputs/cache/test_all_predictions.parquet
(their cached url/text/image branch probabilities, already computed by the
evaluation pipeline), feeds those probabilities through the app's own fusion
and policy code (src/phishlens/fusion.py, src/phishlens/policy.py) using the
installed models/fusion.json, and compares:

  - the fused "learned" score against the evaluation's score_learned column
  - the three-outcome label against the outcome the same policy code gives
    when run directly against the evaluation's cached probabilities

Prints a table and the maximum absolute score difference. Exit code is 0 if
the maximum difference is at or below FLOAT_TOL and every outcome matches,
1 otherwise.

Run from the project root:
    python scripts/check_fusion_consistency.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phishlens import fusion, policy, config  # noqa: E402

CACHE = ROOT / "evaluation" / "outputs" / "cache" / "test_all_predictions.parquet"
N_CASES = 20
FLOAT_TOL = 1e-9


def main() -> int:
    df = pd.read_parquet(CACHE)
    sample = df.sort_values("id").head(N_CASES).reset_index(drop=True)

    cfg = fusion.load_config(config.fusion_path())
    assert cfg.fitted, "models/fusion.json is not marked fitted; installed the wrong file?"

    rows = []
    max_abs_diff = 0.0
    all_outcomes_match = True

    for _, r in sample.iterrows():
        probs = {"url": float(r["p_url"]), "text": float(r["p_text"]), "image": float(r["p_image_head"])}
        result = fusion.fuse_probabilities(probs, cfg, method="learned")
        app_score = result.score
        eval_score = float(r["score_learned"])
        diff = abs(app_score - eval_score)
        max_abs_diff = max(max_abs_diff, diff)

        app_decision = policy.decide(
            app_score, probs, cfg.thresholds["low"], cfg.thresholds["high"], cfg.disagreement_gap
        )
        eval_decision = policy.decide(
            eval_score, probs, cfg.thresholds["low"], cfg.thresholds["high"], cfg.disagreement_gap
        )
        outcome_match = app_decision.outcome == eval_decision.outcome
        all_outcomes_match = all_outcomes_match and outcome_match

        rows.append({
            "id": r["id"],
            "label": int(r["label"]),
            "eval_score_learned": eval_score,
            "app_fused_score": app_score,
            "abs_diff": diff,
            "eval_outcome": eval_decision.outcome,
            "app_outcome": app_decision.outcome,
            "outcome_match": outcome_match,
        })

    out = pd.DataFrame(rows)
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(out.to_string(index=False))
    print()
    print("n_cases:", len(out))
    print("max_abs_score_diff: %.3e" % max_abs_diff)
    print("all_outcomes_match:", all_outcomes_match)
    print("fusion_config_fitted:", cfg.fitted, "method:", cfg.method)
    print("thresholds:", cfg.thresholds, "disagreement_gap:", cfg.disagreement_gap)

    ok = (max_abs_diff <= FLOAT_TOL) and all_outcomes_match
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
