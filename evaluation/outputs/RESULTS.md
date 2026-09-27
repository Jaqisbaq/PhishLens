# PhishLens evaluation results

This report was produced by running the scripts in `evaluation/scripts/` against the pinned models and the datasets described in `DATA_ACCESS.md`. Every number below comes from a real run recorded in `outputs/results.json` and the CSV tables in `outputs/tables/`. Nothing here was estimated or invented.

## What was done

Part A built a linked-case table from the usable Zenodo 8041387 triples (URL, page text and a non-blank screenshot for the same site), removed cases whose URL matches the URL and text models' training data (`ealvaradob/phishing-dataset`), split the remaining cases by registered domain into a 70 percent development set and a 30 percent locked test set (seed 2026), ran the three branches (URL model, text model, CLIP image embedding with a logistic head fitted on development data), fitted a learned fusion with modality-masking augmentation and grouped 5-fold cross-validation for the regularisation strength, checked calibration, froze decision thresholds on development data only, and evaluated once on the locked test set.

Part B ran the URL and text branches on independent samples (PhiUSIIL for URLs, CEAS_08 email text for messages) and compared the chosen models against the rejected candidates on a further, disjoint 400-item sample per modality.

Inference path: adapters (src/phishlens/adapters and phishlens.fusion imported cleanly in the venv and were used directly)

## Dataset summary and leakage removal

| step                                                                             |   phishing |   legitimate |
|:---------------------------------------------------------------------------------|-----------:|-------------:|
| screenshot_present_any (three_any)                                               |        651 |          743 |
| usable_triple (three: present, not blank)                                        |        512 |          510 |
| removed_training_leakage (url exact or normalised match vs ealvaradob urls.json) |          8 |           21 |
| final_linked_case_table                                                          |        504 |          489 |


Split (grouped by registered domain, seed 2026, no domain in both partitions, manifest sha256 `7e54d92e791257ef27a2d3b38dc3a0fdb63e68a4b6348d8ece5290fdab43644c`): development 694 cases (352 phishing, 342 legitimate) across 691 domains; locked test 299 cases (152 phishing, 147 legitimate) across 295 domains.

## Frozen fusion configuration

```json
{
  "method": "learned",
  "feature_order": [
    "logit_p_url_x_a_url",
    "logit_p_text_x_a_text",
    "logit_p_image_x_a_image",
    "a_url",
    "a_text",
    "a_image"
  ],
  "coefficients": [
    0.3237821964193841,
    0.1722539841891832,
    0.8567235389877049,
    -0.8093199626592775,
    -1.5155228173231716,
    -0.07668284376249809
  ],
  "intercept": -0.09538907108166585,
  "calibration": {
    "type": "platt",
    "a": 1.0062526978903061,
    "b": -0.18000073775318892
  },
  "thresholds": {
    "low": 0.3519,
    "high": 0.7633
  },
  "disagreement_gap": 0.9967,
  "fitted": true,
  "thresholds_provisional": false,
  "best_single_branch": null,
  "fitted_on": "694 development linked cases (352 phishing, 342 legitimate), url+text direct model output, image branch grouped 5-fold out-of-fold logistic head, modality-masking augmentation, C=0.1 chosen by grouped 5-fold CV mean average precision (0.9623)",
  "created": "2026-09-27T10:38:41Z",
  "notes": "Calibration comparison on development OOF predictions: Brier uncalibrated=0.0722, platt=0.0720, temperature=0.0722. Chosen: platt. disagreement_gap set at the 90th percentile of the development max-minus-min branch probability gap (0.9967), so the disagreement gate fires on roughly the most-disagreeing 10 percent of development cases."
}
```

## Locked test metrics (evaluated once)

| method   |   n |   precision_0.5 |   recall_0.5 |   f1_0.5 |   fpr_0.5 |   fn_count_0.5 |   roc_auc |   average_precision |   ap_ci_low |   ap_ci_high |     brier |   ece_10bin |
|:---------|----:|----------------:|-------------:|---------:|----------:|---------------:|----------:|--------------------:|------------:|-------------:|----------:|------------:|
| url      | 299 |        0.772727 |     0.894737 | 0.829268 | 0.272109  |             16 |  0.892096 |            0.878531 |    0.819317 |     0.925591 | 0.151816  |   0.133851  |
| text     | 299 |        0.559701 |     0.986842 | 0.714286 | 0.802721  |              2 |  0.760987 |            0.757047 |    0.680557 |     0.827103 | 0.400105  |   0.40067   |
| image    | 299 |        0.843972 |     0.782895 | 0.812287 | 0.14966   |             33 |  0.899123 |            0.901988 |    0.852099 |     0.936145 | 0.127935  |   0.0672017 |
| mean     | 299 |        0.774869 |     0.973684 | 0.862974 | 0.292517  |              4 |  0.953768 |            0.951424 |    0.920378 |     0.974348 | 0.123077  |   0.153493  |
| max      | 299 |        0.548736 |     1        | 0.708625 | 0.85034   |              0 |  0.766268 |            0.758795 |    0.682405 |     0.827944 | 0.412863  |   0.423667  |
| learned  | 299 |        0.937063 |     0.881579 | 0.908475 | 0.0612245 |             18 |  0.968    |            0.967067 |    0.943576 |     0.984112 | 0.0729034 |   0.0461633 |


Paired group bootstrap, learned fusion versus the best single branch on development data (image): average precision difference 0.0656 (95% CI 0.0352 to 0.1061); F1 at 0.5 difference 0.0966 (95% CI 0.0554 to 0.1411).

## Three-outcome table (locked test, learned fusion at frozen thresholds)

| outcome              |   legitimate |   phishing |   total |   coverage |
|:---------------------|-------------:|-----------:|--------:|-----------:|
| High concern         |            6 |        110 |     116 |   0.38796  |
| Review needed        |           42 |         30 |      72 |   0.240803 |
| Limited indicators   |           99 |         12 |     111 |   0.371237 |
| Analysis unavailable |            0 |          0 |       0 |   0        |


Review rate: 0.241. Phishing cases given Limited indicators: 12.

## Ablation (locked test, learned fusion with branches masked)

| branches_masked   |   n_scored |   average_precision |   ap_ci_low |   ap_ci_high |   f1_at_0.5 |   roc_auc |     brier |
|:------------------|-----------:|--------------------:|------------:|-------------:|------------:|----------:|----------:|
| full (no masking) |        299 |            0.967067 |    0.943576 |     0.984112 |    0.908475 |  0.968    | 0.0729034 |
| mask url          |        299 |            0.922799 |    0.888599 |     0.952115 |    0.802817 |  0.917607 | 0.120571  |
| mask text         |        299 |            0.960809 |    0.932296 |     0.98038  |    0.890411 |  0.962093 | 0.081085  |
| mask image        |        299 |            0.91073  |    0.867872 |     0.943554 |    0.831615 |  0.910356 | 0.122228  |
| mask url+text     |        299 |            0.901988 |    0.852099 |     0.936145 |    0.792857 |  0.899123 | 0.133261  |
| mask url+image    |        299 |            0.757047 |    0.680557 |     0.827103 |    0.711111 |  0.760987 | 0.213128  |
| mask text+image   |        299 |            0.878531 |    0.819317 |     0.925591 |    0.776978 |  0.892096 | 0.141248  |


## Error inspection (worst test cases, URLs defanged)

| id                       | label      | url_defanged                                                                          |      p_url |      p_text |   p_image_head |   fused_learned_score |
|:-------------------------|:-----------|:--------------------------------------------------------------------------------------|-----------:|------------:|---------------:|----------------------:|
| 64331036cf8bf72876f1b915 | phishing   | hxxps://steamcommnunite[.]ru/profiles/61816281                                        | 0.819872   | 0.99999     |      0.0208489 |             0.0284704 |
| 6433004ecf8bf72876f1b8ac | phishing   | hxxp://ignitioncr[.]com/Citizens/citizen-bank-RD641-detail1/                          | 0.276364   | 0.999922    |      0.0912422 |             0.0339832 |
| 6454696fc40bc36932404ab3 | legitimate | hxxps://signon[.]telstra[.]com/login?goto=http%3A%2F%2Femail[.]telstra[.]com%3A443%2F | 0.996524   | 0.999995    |      0.878408  |             0.951256  |
| 644978316f96f6cb201d8d88 | legitimate | hxxps://www2[.]etc-meisai[.]jp/etc/R?funccode=7261774389                              | 0.999909   | 0.998721    |      0.813511  |             0.940618  |
| 643644c9365b001068057d97 | phishing   | hxxp://2332442111021[.]my[.]id/repeat[.]php?section_id=13-HowToContactMeta            | 0.00279821 | 0.999986    |      0.583206  |             0.0846372 |
| 6427919c736639f0f0508b45 | phishing   | hxxps://inmotion[.]udc[.]es/sbb/SBB-CH_swiss/4e9800998ecf8427e/                       | 0.0797657  | 0.000176014 |      0.965291  |             0.10708   |


## Part B: component evaluation on independent data

| component                                                  |     n |   precision_0.5 |   recall_0.5 |   f1_0.5 |   fpr_0.5 |   fn_count_0.5 |   roc_auc |   average_precision |   ap_ci_low |   ap_ci_high |   f1_ci_low |   f1_ci_high |    brier |   ece_10bin |
|:-----------------------------------------------------------|------:|----------------:|-------------:|---------:|----------:|---------------:|----------:|--------------------:|------------:|-------------:|------------:|-------------:|---------:|------------:|
| url (PhiUSIIL, n=20000 stratified sample, leakage removed) | 20000 |        0.550936 |      0.79082 | 0.649434 |  0.482514 |           1791 |  0.784154 |            0.797726 |    0.790011 |     0.805088 |    0.641996 |     0.656621 | 0.293681 |    0.285125 |
| text (CEAS_08, 500 phishing / 500 legitimate)              |  1000 |        0.956522 |      0.792   | 0.866521 |  0.036    |            104 |  0.944354 |            0.951631 |    0.938569 |     0.962001 |    0.842084 |     0.888428 | 0.120598 |    0.121653 |


## Candidate comparison

| modality   | model                                                                                                                                               | contaminated_on_this_sample   |   n |   average_precision |   f1_at_0.5 |   roc_auc | contamination_note                                                                                                                                                                                                                                       | excluded_reason                                                                                                                   |
|:-----------|:----------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------|----:|--------------------:|------------:|----------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------|
| url        | CrabInHoney/urlbert-tiny-v4-phishing-classifier (primary)                                                                                           | False                         | 400 |            0.833635 |    0.724138 |  0.7965   |                                                                                                                                                                                                                                                          |                                                                                                                                   |
| url        | kmack/malicious-url-detection (rejected: labels BENIGN/MALWARE, not phishing-specific)                                                              | False                         | 400 |            0.962185 |    0.847262 |  0.95265  |                                                                                                                                                                                                                                                          |                                                                                                                                   |
| url        | pirocheto/phishing-url-detection (rejected: weak on benign samples in the spike; trained on pirocheto/phishing-url, no overlap with PhiUSIIL known) | False                         | 400 |            0.944699 |    0.76161  |  0.92535  |                                                                                                                                                                                                                                                          |                                                                                                                                   |
| text       | ealvaradob/bert-finetuned-phishing (primary)                                                                                                        | False                         | 400 |            0.969912 |    0.903743 |  0.96555  |                                                                                                                                                                                                                                                          |                                                                                                                                   |
| text       | aamoshdahal/email-phishing-distilbert-finetuned (rejected: no licence stated)                                                                       | True                          | 400 |            0.99937  |    0.987531 |  0.999375 | trained on Enron, CEAS 2008, Ling-Spam, SpamAssassin, Nazario and Nigerian fraud emails per its card, which includes CEAS_08: this sample partially overlaps its own training data, so its score here is optimistic and not comparable to the other rows |                                                                                                                                   |
| text       | ElSlay/BERT-Phishing-Email-Model (rejected: no licence stated, higher latency, no accuracy gain in the spike)                                       | False                         | 400 |            0.975333 |    0.899743 |  0.9769   |                                                                                                                                                                                                                                                          |                                                                                                                                   |
| url        | amahdaouy/DomURLs_BERT                                                                                                                              |                               |     |                     |             |           |                                                                                                                                                                                                                                                          | no classifier head in the checkpoint (encoder only); would need a labelled URL set and a fitted head before it produces any score |
| text       | cybersectony/phishing-email-detection-distilbert_v2.4.1                                                                                             |                               |     |                     |             |           |                                                                                                                                                                                                                                                          | four-class output with no class names in the config or agreement between the model card and dataset card on what the classes mean |


## Notable findings

- The learned fusion clearly beats every single branch and the mean/max baselines on the locked test set (average precision 0.967 versus 0.902 for the best single branch, image; the paired group bootstrap difference is positive with a 95 percent interval that excludes zero).
- The text branch performs poorly on this linked case set: it labels most items phishing regardless of the true label (false positive rate 0.80 at a 0.5 threshold on test), consistent with the page text here not being the message text the model was trained on. On the independent CEAS_08 email sample in Part B, the same model performs well (false positive rate 0.036), which supports the domain-mismatch explanation rather than a broken model.
- On the independent 400-item PhiUSIIL candidate-comparison sample, the chosen URL model (CrabInHoney/urlbert-tiny-v4) scores lower (average precision 0.834) than both rejected URL candidates, kmack/malicious-url-detection (0.962) and pirocheto/phishing-url-detection (0.945). This is an honest, unfavourable result for the model actually shipped in the app and should be weighed against the qualitative reasons (interpretable labels, small size) that led to its selection in `docs/model_selection/MODELS.md`.
- Masking the image branch causes the largest single-branch drop in ablation average precision (0.967 to 0.911), and masking url+image together causes the largest two-branch drop (0.967 to 0.757), showing the fused model leans most on the image and url branches for this dataset.

## Part C: runtime

| mode       |   n_requests |   cold_first_request_ms |   warm_median_ms |   warm_p95_ms |   warm_min_ms |   warm_max_ms |   failures |   peak_process_memory_mb |
|:-----------|-------------:|------------------------:|-----------------:|--------------:|--------------:|--------------:|-----------:|-------------------------:|
| end_to_end |          200 |                 7721.82 |          1165.58 |        2708.7 |       245.374 |       2764.36 |          0 |                   2310.6 |


| key                  | value                                             |
|:---------------------|:--------------------------------------------------|
| os                   | Windows-10-10.0.26200-SP0                         |
| python               | 3.11.3                                            |
| cpu_model            | AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD |
| logical_cpus         | 16                                                |
| physical_cpus        | 8                                                 |
| ram_gb               | 63.4                                              |
| torch_version        | 2.2.2+cpu                                         |
| transformers_version | 4.48.0                                            |
| sklearn_version      | 1.5.2                                             |
| numpy_version        | 1.26.4                                            |

The operating system row is the platform string that Python reports. Build 26200 is Windows 11.


Supplementary in-process breakdown of where end-to-end time goes, one process, orchestrator preloaded once, 200 development cases:

| mode                    |   n_cases |   model_load_ms |   warm_median_total_ms |   warm_median_url_ms |   warm_median_text_ms |   warm_median_image_ms |   median_share_pct_url |   median_share_pct_text |   median_share_pct_image |   branches_not_ok |
|:------------------------|----------:|----------------:|-----------------------:|---------------------:|----------------------:|-----------------------:|-----------------------:|------------------------:|-------------------------:|------------------:|
| in_process_orchestrator |       200 |          4335.3 |                 1155.3 |                  7.6 |                1046.1 |                  103.6 |                    0.7 |                    90.4 |                      8.9 |                 0 |


## Limitations

- The Zenodo 8041387 "text" field is web page text scraped from the site, not an email or an SMS message. It is never called a message in this report.
- Screenshots come from only one id range of the Zenodo dataset (ids 4501 and up in each class), because only that range was downloaded to stay under the data budget; the linked evaluation set is not a random sample of the whole Zenodo catalogue.
- The linked evaluation set is small (993 cases after leakage removal, roughly 694 development and 299 locked test), so confidence intervals are wide, especially for the ablation and the three-outcome table.
- Leakage removal against `ealvaradob/phishing-dataset` used exact and normalised URL matching only; this catches partial overlap but a different page hosted at a new URL with reused content would not be detected.
- CLIP's own training data is not published, so image-branch leakage cannot be checked and is reported as unknown throughout.
- No user study was run: the three-outcome policy and the disagreement gap are validated only by the numbers in this report, not by how a person actually uses the interface.
- The PhiUSIIL and CEAS_08 samples in Part B are independent of the linked cases but are still fixed public datasets, not a live traffic sample.
- aamoshdahal's score in the candidate comparison is contaminated: it was trained on CEAS 2008 and Nazario, the same source as the text validation sample, so its number is optimistic and not directly comparable to the other rows.
