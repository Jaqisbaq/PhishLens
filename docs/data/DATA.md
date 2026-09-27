# PhishLens evaluation data: status and recommendations

Date accessed: 2026-09-27. All counts below come from loading the files on disk in `data/raw/` (see `PROVENANCE.csv` for file-level sha256, size and publisher checksum match). Nothing outside the project's data folder was touched. Total size on disk: 1,168,306,256 bytes (about 1.11 GB), under the 5 GB budget.

## 1. Linked data answer (highest priority)

**Yes, genuinely linked multimodal cases exist**, in `zenodo_8041387_phishing_website`. Each row in `phishing.csv` / `not-phishing.csv` has a unique `_id` that is also the screenshot filename and the folder key inside the dataset's zip archives. The link is real (same crawl of the same site), not a same-label pairing.

- Linking key: `_id` (site id), present in `phishing.csv`/`not-phishing.csv`, in `zip_listings/*.members.csv`, and as the screenshot filename `<_id>.jpg`.
- Modalities linked per item: **URL + page text (`features.text` column) + screenshot**. HTML (`features.html`) and CSS are also present under the same key, so 4 modalities are technically linked, but only URL+text+screenshot matter for PhishLens.
- Full metadata (URL, text, HTML) is complete for all 10,395 rows (5,151 phishing + 5,244 not-phishing), confirmed by loading both CSVs (row counts above) and matching Zenodo's published md5 for both files.
- Screenshots are only downloaded for a subset: the two dataset zips were fetched by ID-range chunks via HTTP range requests (not the full 200MB-1GB zips) to stay in budget. Only the last chunk of each class was pulled: phishing ids in `4501-5151`, not-phishing ids in `4501-5244`.
- Verified triple counts, from `_logs/zenodo_8041387_linked_check.csv` (1,395 rows checked, one row per screenshot obtained):
  - **Screenshot file present (`three_any`)**: phishing = 651, not-phishing = 743.
  - **Screenshot file present AND not blank/corrupt (`three`, the usable linked triple)**: **phishing = 512, not-phishing = 510**.
- These 1,022 usable linked triples (512 phishing, 510 legitimate) are the full linked evaluation set obtainable under the size budget. The other ~9,373 rows have URL+text but no screenshot on disk (not downloaded, to stay under 5 GB; they could be fetched later from `zip_listings` chunk `0-4500` if a bigger budget is approved).

## 2. Leakage check (vs `ealvaradob_phishing_dataset` urls.json / texts.json)

| Candidate set | Exact URL/text overlap | Host/prefix overlap | Verdict |
|---|---|---|---|
| uci_phiusiil (235,795 URLs) | 3,019 (1.28%) | 14,825 hosts (6.29%) | partial, usable |
| pirocheto_phishing_url | 11,430 of 11,430 (100%) | 100% | EXCLUDED, fully overlapping |
| shresthsamyak_screenshots (7,465 with URL) | 33 (0.44%) | 611 hosts (8.18%) | partial, usable |
| zenodo_8041387 phishing.csv | 70 (1.36%) | 153 hosts (2.97%) | partial, usable |
| zenodo_8041387 not-phishing.csv | 146 (2.78%) | 832 hosts (15.87%) | partial, usable |
| zefang_liu_phishing_email | 18,098 of 18,631 (97.14%) | - | EXCLUDED, same Kaggle/Enron source as training |
| uci_sms_spam_collection | 2,102 of 5,572 (37.72%) | - | EXCLUDED unless overlap rows removed |
| zenodo_8339691 Nazario | 0 (0%) | - | clean, usable |
| zenodo_8339691 CEAS_08 | 0 (0%) | - | clean, usable |
| zenodo_8339691 Nigerian_Fraud | 0-0.03% | - | clean, but advance-fee fraud not credential phishing |
| zenodo_8339691 SpamAssasin | up to 78.45% | - | EXCLUDED, overlaps training |
| zenodo_8339691 Enron | up to 33.9% (subject+body) | - | EXCLUDED, overlaps training |
| zenodo_8339691 Ling | up to 90.14% | - | EXCLUDED, overlaps training |

CLIP (`openai/clip-vit-base-patch32`) training data is not released, so screenshot overlap cannot be checked for any image set; marked "unknown" throughout.

## 3. Verification summary

All files listed in `PROVENANCE.csv` were loaded (pandas/csv/parquet/zipfile/PIL) without error; publisher md5 checked and matched for both Zenodo records. A random sample of screenshot JPEGs opened and verified with PIL. No corrupt archives. `zenodo_8041387` screenshots use `csv.field_size_limit` workaround because `features.text`/`features.html` fields exceed the Python default limit; this is expected, not a data problem.

## 4. Recommended dataset per modality (CPU-only overnight run)

- **A. URL**: `uci_phiusiil`, 235,795 rows (134,850 legit / 100,945 phishing), 1.28% exact leakage. Sample 5,000-10,000 stratified rows.
- **B. Text**: `zenodo_8339691_email_curated` CEAS_08 (39,154 rows, 21,842 phishing/17,312 legit, 0% leakage) and Nazario (1,565 phishing, 0% leakage, phishing-only so pair with CEAS_08 legit rows).
- **C. Screenshot**: `zenodo_8041387_phishing_website` screenshots (651 phishing, 743 not-phishing files on disk) or `shresthsamyak_screenshots` (446 phishing, 500-7,924 legitimate) as a secondary source.
- **D. Linked**: `zenodo_8041387_phishing_website`, 512 phishing + 510 legitimate usable URL+text+screenshot triples (see section 1).

## Blockers / concerns

- `shresthsamyak_screenshots` licence: card names no explicit licence; third-party brand screenshots, ethical/legal caution advised.
- `zenodo_8041387` screenshots cover only ~13% (phishing) and ~14% (not-phishing) of the full id range; expanding requires downloading earlier zip chunks (`0-4500`), each several hundred MB, which would approach or exceed the 5 GB budget.
- No Kaggle sources were used (login required, excluded per spec).
