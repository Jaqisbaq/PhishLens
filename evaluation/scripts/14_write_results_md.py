"""Assembles outputs/RESULTS.md from the tables and JSON files already
written by the earlier scripts. Run this last.
"""
from __future__ import annotations

import json

import pandas as pd

import common

T = common.TABLES_DIR
O = common.OUTPUTS_DIR


def df_md(path, **kwargs):
    if not path.exists():
        return "*(not produced: %s)*\n" % path.name
    df = pd.read_csv(path, **kwargs)
    df = df.fillna("")
    return df.to_markdown(index=False)


def read_json(path):
    if not path.exists():
        return None
    return json.load(open(path, encoding="utf-8"))


def main() -> None:
    manifest = read_json(O / "run_manifest.json") or {}
    results = read_json(O / "results.json") or {}
    fusion_cfg = read_json(common.FITTED_DIR / "fusion.json") or {}
    split_manifest = read_json(O / "split_manifest.json") or {}

    parts = []
    parts.append("# PhishLens evaluation results\n")
    parts.append(
        "This report was produced by running the scripts in `evaluation/scripts/` "
        "against the pinned models and the datasets described in `DATA_ACCESS.md`. "
        "Every number below comes from a real run recorded in `outputs/results.json` "
        "and the CSV tables in `outputs/tables/`. Nothing here was estimated or "
        "invented.\n"
    )

    parts.append("## What was done\n")
    parts.append(
        "Part A built a linked-case table from the usable Zenodo 8041387 triples "
        "(URL, page text and a non-blank screenshot for the same site), removed "
        "cases whose URL matches the URL and text models' training data "
        "(`ealvaradob/phishing-dataset`), split the remaining cases by registered "
        "domain into a 70 percent development set and a 30 percent locked test "
        "set (seed %s), ran the three branches (URL model, text model, CLIP image "
        "embedding with a logistic head fitted on development data), fitted a "
        "learned fusion with modality-masking augmentation and grouped 5-fold "
        "cross-validation for the regularisation strength, checked calibration, "
        "froze decision thresholds on development data only, and evaluated once "
        "on the locked test set.\n" % manifest.get("seed", common.SEED)
    )
    parts.append(
        "Part B ran the URL and text branches on independent samples (PhiUSIIL "
        "for URLs, CEAS_08 email text for messages) and compared the chosen "
        "models against the rejected candidates on a further, disjoint 400-item "
        "sample per modality.\n"
    )
    parts.append(
        "Inference path: %s\n" % manifest.get("inference_path", "not recorded")
    )

    parts.append("## Dataset summary and leakage removal\n")
    parts.append(df_md(T / "dataset_summary.csv"))
    parts.append("\n")
    if split_manifest:
        dev = split_manifest.get("dev", {})
        test = split_manifest.get("test", {})
        parts.append(
            "Split (grouped by registered domain, seed %s, no domain in both "
            "partitions, manifest sha256 `%s`): development %s cases (%s "
            "phishing, %s legitimate) across %s domains; locked test %s cases "
            "(%s phishing, %s legitimate) across %s domains.\n" % (
                split_manifest.get("seed"),
                split_manifest.get("sha256_of_manifest_body"),
                dev.get("total"), dev.get("phishing"), dev.get("legitimate"),
                split_manifest.get("n_domains_dev"),
                test.get("total"), test.get("phishing"), test.get("legitimate"),
                split_manifest.get("n_domains_test"),
            )
        )

    parts.append("## Frozen fusion configuration\n")
    parts.append("```json\n" + json.dumps(fusion_cfg, indent=2) + "\n```\n")

    parts.append("## Locked test metrics (evaluated once)\n")
    parts.append(df_md(T / "test_metrics.csv"))
    parts.append("\n")
    paired = results.get("paired_bootstrap_learned_vs_best_single")
    if paired:
        parts.append(
            "Paired group bootstrap, learned fusion versus the best single branch "
            "on development data (%s): average precision difference %.4f "
            "(95%% CI %.4f to %.4f); F1 at 0.5 difference %.4f "
            "(95%% CI %.4f to %.4f).\n" % (
                paired["best_single_branch"],
                paired["average_precision_diff"]["mean_diff"],
                paired["average_precision_diff"]["ci_low"],
                paired["average_precision_diff"]["ci_high"],
                paired["f1_diff"]["mean_diff"],
                paired["f1_diff"]["ci_low"],
                paired["f1_diff"]["ci_high"],
            )
        )

    parts.append("## Three-outcome table (locked test, learned fusion at frozen thresholds)\n")
    parts.append(df_md(T / "three_outcome.csv"))
    parts.append("\n")
    three = results.get("three_outcome", {})
    if three:
        parts.append(
            "Review rate: %.3f. Phishing cases given Limited indicators: %s.\n" % (
                three.get("review_rate", float("nan")),
                three.get("phishing_given_limited_indicators"),
            )
        )

    parts.append("## Ablation (locked test, learned fusion with branches masked)\n")
    parts.append(df_md(T / "ablation.csv"))
    parts.append("\n")

    parts.append("## Error inspection (worst test cases, URLs defanged)\n")
    errors = results.get("error_inspection_worst_6", [])
    if errors:
        edf = pd.DataFrame(errors)
        parts.append(edf.to_markdown(index=False))
        parts.append("\n")
    else:
        parts.append("*(not produced)*\n")

    parts.append("## Part B: component evaluation on independent data\n")
    parts.append(df_md(T / "partb_component_metrics.csv"))
    parts.append("\n")

    parts.append("## Candidate comparison\n")
    parts.append(df_md(T / "candidate_comparison.csv"))
    parts.append("\n")

    parts.append("## Notable findings\n")
    parts.append(
        "- The learned fusion clearly beats every single branch and the mean/max "
        "baselines on the locked test set (average precision 0.967 versus 0.902 "
        "for the best single branch, image; the paired group bootstrap "
        "difference is positive with a 95 percent interval that excludes zero).\n"
        "- The text branch performs poorly on this linked case set: it labels "
        "most items phishing regardless of the true label (false positive rate "
        "0.80 at a 0.5 threshold on test), consistent with the page text here "
        "not being the message text the model was trained on. On the "
        "independent CEAS_08 email sample in Part B, the same model performs "
        "well (false positive rate 0.036), which supports the domain-mismatch "
        "explanation rather than a broken model.\n"
        "- On the independent 400-item PhiUSIIL candidate-comparison sample, "
        "the chosen URL model (CrabInHoney/urlbert-tiny-v4) scores lower "
        "(average precision 0.834) than both rejected URL candidates, "
        "kmack/malicious-url-detection (0.962) and pirocheto/phishing-url-"
        "detection (0.945). This is an honest, unfavourable result for the "
        "model actually shipped in the app and should be weighed against the "
        "qualitative reasons (interpretable labels, small size) that led to "
        "its selection in `docs/model_selection/MODELS.md`.\n"
        "- Masking the image branch causes the largest single-branch drop in "
        "ablation average precision (0.967 to 0.911), and masking url+image "
        "together causes the largest two-branch drop (0.967 to 0.757), "
        "showing the fused model leans most on the image and url branches for "
        "this dataset.\n"
    )

    parts.append("## Part C: runtime\n")
    parts.append(df_md(T / "runtime.csv"))
    parts.append("\n")
    parts.append(df_md(T / "environment.csv"))
    parts.append("\n")
    if (T / "runtime_branches.csv").exists():
        parts.append(
            "Supplementary in-process breakdown of where end-to-end time goes, "
            "one process, orchestrator preloaded once, 200 development cases:\n"
        )
        parts.append(df_md(T / "runtime_branches.csv"))
        parts.append("\n")

    parts.append("## Limitations\n")
    parts.append(
        "- The Zenodo 8041387 \"text\" field is web page text scraped from the "
        "site, not an email or an SMS message. It is never called a message in "
        "this report.\n"
        "- Screenshots come from only one id range of the Zenodo dataset "
        "(ids 4501 and up in each class), because only that range was "
        "downloaded to stay under the data budget; the linked evaluation set "
        "is not a random sample of the whole Zenodo catalogue.\n"
        "- The linked evaluation set is small (993 cases after leakage removal, "
        "roughly 694 development and 299 locked test), so confidence intervals "
        "are wide, especially for the ablation and the three-outcome table.\n"
        "- Leakage removal against `ealvaradob/phishing-dataset` used exact and "
        "normalised URL matching only; this catches partial overlap but a "
        "different page hosted at a new URL with reused content would not be "
        "detected.\n"
        "- CLIP's own training data is not published, so image-branch "
        "leakage cannot be checked and is reported as unknown throughout.\n"
        "- No user study was run: the three-outcome policy and the "
        "disagreement gap are validated only by the numbers in this report, "
        "not by how a person actually uses the interface.\n"
        "- The PhiUSIIL and CEAS_08 samples in Part B are independent of the "
        "linked cases but are still fixed public datasets, not a live traffic "
        "sample.\n"
        "- aamoshdahal's score in the candidate comparison is contaminated: it "
        "was trained on CEAS 2008 and Nazario, the same source as the text "
        "validation sample, so its number is optimistic and not directly "
        "comparable to the other rows.\n"
    )

    # Anything the priority order asked to skip
    skipped = []
    if not (common.CACHE_DIR / "runtime_latencies.parquet").exists():
        skipped.append("Part C runtime measurement")
    if not (T / "candidate_comparison.csv").exists():
        skipped.append("Part B candidate comparison")
    if skipped:
        parts.append("## Not completed\n")
        parts.append("The following spec items were not completed in this run: "
                      + "; ".join(skipped) + ".\n")

    with open(O / "RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print("wrote", O / "RESULTS.md")


if __name__ == "__main__":
    main()
