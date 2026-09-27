# Data access

No data files are copied into this project. The evaluation scripts read
directly from a `data` folder placed beside the repository folder, which is not
part of this repository. To reproduce the evaluation, obtain each dataset
below and place it under `data/raw/<name>/` exactly as named in
`docs/data/PROVENANCE.csv`, then run the scripts in `evaluation/scripts/`
in numeric order.

## Datasets used

| Role | Dataset | Source | Licence | Used for |
|---|---|---|---|---|
| Linked cases (Part A, main result) | `zenodo_8041387_phishing_website` (`phishing.csv`, `not-phishing.csv`, `screenshots/`) | https://zenodo.org/records/8041387 (doi 10.5281/zenodo.8041387) | CC-BY-4.0 | URL, page text and screenshot triples |
| URL component (Part B) | `uci_phiusiil` | https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset | CC-BY-4.0 | independent URL sample |
| Text component (Part B) | `zenodo_8339691_email_curated` (`CEAS_08.csv`) | https://zenodo.org/records/8339691 (doi 10.5281/zenodo.8339691) | CC-BY-4.0 (record licence; underlying corpus terms not verified) | independent message-text sample |
| Leakage reference, never evaluated on | `ealvaradob_phishing_dataset` (`urls.json`) | https://huggingface.co/datasets/ealvaradob/phishing-dataset | Apache-2.0 | removing cases that overlap the URL and text models' training data |

Exact file names, byte sizes and sha256 hashes for every file are recorded
in `docs/data/PROVENANCE.csv`. Full counts, the linked-triple check and the
overlap percentages are recorded in `docs/data/DATA.md` and
`docs/data/logs/` (`zenodo_8041387_linked_check.csv`, `overlap_urls.json`,
`overlap_texts.json`).

## Directory layout expected by the scripts

```
work/
  data/
    raw/
      zenodo_8041387_phishing_website/
        phishing.csv
        not-phishing.csv
        screenshots/phishing/<_id>.jpg
        screenshots/not-phishing/<_id>.jpg
      uci_phiusiil/phiusiil_phishing_url_dataset.zip
      zenodo_8339691_email_curated/CEAS_08.csv
      ealvaradob_phishing_dataset/urls.json
    _logs/
      zenodo_8041387_linked_check.csv
      overlap_urls.json
      overlap_texts.json
  phishlens/
    evaluation/   <- this folder
```

## Models

All models are pulled from the Hugging Face Hub at the pinned revisions
listed in `work/specs/APP_SPEC.md` and `outputs/run_manifest.json`, and
must be present in the local Hugging Face cache before running with
`HF_HUB_OFFLINE=1` (no dataset URL or model card page is ever fetched by
these scripts; model weights come from the Hub's model repositories, not
from the evaluation datasets).

## Running

```
cd evaluation/scripts
python 01_build_case_table.py
python 02_split.py
python 03_run_url_text.py url
python 03_run_url_text.py text
python 04_run_image.py
python 05_fit_image_head.py
python 06_fit_fusion.py
python 09_evaluate_test.py
python 07_part_b_url.py
python 08_part_b_text.py
python 11_part_b_metrics.py
python 10_part_b_candidates.py
python fig1_case_counts.py
python fig2_3_4.py
python fig5_ablation.py
python fig6_latency.py
python fig7_candidates.py
```

Every script checks its output cache before recomputing, so a run can be
interrupted and restarted. `python` here means the interpreter in
`.venv` (Python 3.11), never `python3`.
