# PhishLens model feasibility spike

Run date: 2026-09-27. Folder: `docs/model_selection`. Machine readable numbers: `results.json`.

## Test conditions

- Windows 11, Python 3.11.3, torch 2.2.2+cpu, transformers 4.48.0, onnxruntime 1.24.4 (already installed). No package was installed, upgraded or downgraded. No venv was needed.
- CPU only, 16 logical CPUs, torch used 8 threads (its default).
- Each candidate ran in its own fresh process, offline (`HF_HUB_OFFLINE=1`) after download.
- Warm latency is the median of 30 single item runs (batch size 1) on short inputs. Long input latency is the median of 20 runs at the 512 token limit.
- Model files were already in the operating system file cache (freshly downloaded), so a load after a reboot will be slower than the load times below. That case was not measured.
- Inputs were 6 hand written messages, 6 hand written URL strings (reserved `example.*` hosts and the 192.0.2.0/24 documentation range) and 4 synthetic screenshots drawn with Pillow. No URL was fetched, resolved or opened.
- The sample inputs are a smoke test, not an evaluation. "Expected" is a hand judgement. Behaviour on 6 inputs is indicative only.

## a. Candidate table

Size is the bytes actually downloaded for inference (TensorFlow, Flax, notebook, `training_args.bin` and `model.pkl` files were skipped on purpose). Load is `from_pretrained` for tokenizer plus model. Library import time is separate (see section d).

| # | Branch | Repo id | Revision | Licence (card metadata) | Size MB | Params | Trained on (per model card) | Labels (id2label in config) | Load ms | Cold first ms | Warm median ms | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Text | ealvaradob/bert-finetuned-phishing | fa8fb73a007174c410ab7160d4e4c6e6b8d998d4 | apache-2.0 | 1341.7 | 335,143,938 | ealvaradob/phishing-dataset (combined: URLs, SMS, emails, website HTML) | 0 benign, 1 phishing | 210.5 | 185.7 | 143.9 short, 2374.2 at 512 tokens | WORKS |
| 2 | Text | cybersectony/phishing-email-detection-distilbert_v2.4.1 | 6724dec17360ab1f6450553f1d950404f66a5dde | apache-2.0 | 268.8 | 66,956,548 | cybersectony/PhishingEmailDetectionv2.0 (22,644 emails and 177,356 URLs) | LABEL_0 to LABEL_3, unnamed in config | 99.0 | 32.8 | 22.7 short, 396.1 at 512 tokens | WORKS |
| 3 | Text | aamoshdahal/email-phishing-distilbert-finetuned | f9d211125fcb6e85cf73630957667bc2a6e93ed3 | none stated | 268.1 | 66,955,010 | Kaggle phishing email dataset (Enron, CEAS 2008, Ling-Spam, SpamAssassin, Nazario, Nigerian fraud). Card says it was evaluated on cybersectony/PhishingEmailDetectionv2.0 | LABEL_0, LABEL_1 (card: 0 legitimate, 1 phishing) | 171.8 | 46.3 | 25.9 short, 372.3 at 512 tokens | WORKS |
| 4 | Text | ElSlay/BERT-Phishing-Email-Model | 708d260869707ffdb0227c3c838ce957a736c9d6 | none stated | 438.2 | 109,483,778 | zefang-liu/phishing-email-dataset | LABEL_0, LABEL_1 (card: 0 legitimate, 1 phishing) | 167.6 | 68.7 | 50.2 short, 772.9 at 512 tokens | WORKS |
| 5 | URL | CrabInHoney/urlbert-tiny-v4-phishing-classifier | fd962a5cd04e20ceed46aa8a6e31a2b40d4c8916 | apache-2.0 | 14.8 | 3,686,210 | ealvaradob/phishing-dataset, urls.json only | LABEL_0, LABEL_1 (card: 0 good, 1 phishing) | 100.1 | 13.2 | 7.7 | WORKS |
| 6 | URL | amahdaouy/DomURLs_BERT | ba5fa28ae1361f8115630e252ac49535ffe3bab1 | none stated | 443.8 | 110,617,344 | Masked language modelling on a multilingual corpus of URLs, domain names and DGA data. Training data section of the card is empty | none, no classifier in checkpoint | 387.1 | 62.7 | 36.0 (embedding only) | NEEDS HEAD |
| 7 | URL | ealvaradob/bert-phishing-url | 1a87f6979f682c1cf0fea476048072f0c5b7c2d0 | apache-2.0 | 1341.6 | 335,143,938 | ealvaradob/phishing-dataset (URL samples) | 0 benign, 1 phishing | 177.5 | 206.6 | 139.7 | WORKS |
| 8 | URL | pirocheto/phishing-url-detection | 44f3b19f705b52532e0aadf3d0d15dd892b8a2fb | mit | 23.6 (ONNX only) | not computed (linear SVM in ONNX, not a neural network) | pirocheto/phishing-url (11,430 URLs, CC-BY-4.0) | output column 0 legitimate, column 1 phishing | 610.2 | 2.6 | 1.9 | WORKS |
| 9 | URL | kmack/malicious-url-detection | 258499831602e1aea6c1f00e8483b820dd14b391 | apache-2.0 | 268.8 | 66,955,010 | kmack/Phishing_urls | 0 BENIGN, 1 MALWARE | 105.4 | 37.5 | 22.4 | WORKS |
| 10 | Image | openai/clip-vit-base-patch32 | 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 | none in card metadata | 608.9 | 151,277,313 (vision tower 87,849,216, text tower 63,428,096) | Public image and caption pairs crawled from the internet plus datasets such as YFCC100M. Not phishing specific | none (zero-shot prompts or a fitted head) | 365.1 | 184.6 | 100.6 image embedding, 123.1 full zero-shot | WORKS |

None of the 10 repos is gated. All 10 loaded and ran. No torch or tokenizer incompatibility was hit.

Peak process memory (Windows peak working set, includes roughly 610 MB of torch plus transformers import baseline):

| Repo | After load MB | Peak MB |
|---|---|---|
| ealvaradob/bert-finetuned-phishing | 612.3 | 1883.9 |
| cybersectony distilbert v2.4.1 | 610.2 | 827.7 |
| aamoshdahal distilbert | 615.9 | 829.5 |
| ElSlay BERT base | 615.1 | 999.3 |
| CrabInHoney urlbert-tiny-v4 | 604.3 | 628.0 |
| amahdaouy/DomURLs_BERT | 636.8 | 999.2 |
| ealvaradob/bert-phishing-url | 614.4 | 1842.3 |
| pirocheto ONNX | 462.0 | 462.0 |
| kmack distilbert | 611.3 | 799.6 |
| openai/clip-vit-base-patch32 | 650.1 | 1182.4 |

Memory with all three chosen models loaded in one process was not measured.

## b. Sample outputs (real numbers from the runs)

### Text branch, P(phishing) per message

For candidate 2 the value shown is the probability of LABEL_1, which the dataset card names `phishing_email`.

| Message | Expected | 1 ealvaradob BERT large | 2 cybersectony | 3 aamoshdahal | 4 ElSlay |
|---|---|---|---|---|---|
| Lunch at 1pm tomorrow? | benign | 0.0000 | 0.0000 | 0.0010 | 0.0017 |
| Hi team, the meeting notes from Monday are attached. Let me know if I missed anything. | benign | 0.0000 | 0.0000 | 0.0018 | 0.0003 |
| Reminder: your dentist appointment is on Friday at 3pm. Reply YES to confirm. | benign | 0.0001 | 0.9839 | 0.0923 | 0.0559 |
| Your parcel is held, confirm your card details at the link below | phishing | 0.9998 | 0.9999 | 0.9116 | 0.9829 |
| URGENT: your account has been suspended. Verify your password within 24 hours to avoid permanent closure. | phishing | 1.0000 | 1.0000 | 0.9994 | 0.9962 |
| Congratulations! You have won a $1000 gift card. Reply with your bank details to claim your prize. | phishing | 1.0000 | 1.0000 | 0.9988 | 0.9978 |

Side check, text models fed the 6 URL strings:

| URL string | Expected | 1 ealvaradob P(phishing) | 2 cybersectony top label | 3 aamoshdahal P(phishing) | 4 ElSlay P(phishing) |
|---|---|---|---|---|---|
| https://www.example.com/login | benign | 0.0000 | LABEL_3 at 1.0000 | 0.0264 | 0.5742 |
| https://www.example.org/docs/getting-started.html | benign | 0.0000 | LABEL_3 at 1.0000 | 0.0097 | 0.0282 |
| https://accounts.example.org/signin | benign | 0.0001 | LABEL_3 at 1.0000 | 0.7625 | 0.7959 |
| http://secure-paypa1-account-verify.example.net/update | phishing | 1.0000 | LABEL_3 at 1.0000 | 0.9996 | 0.9992 |
| http://192.0.2.10/bank/login.php?user=verify&session=8813 | phishing | 1.0000 | LABEL_3 at 1.0000 | 0.7642 | 0.9983 |
| http://example-bank-security-alert.example.com/confirm/account/password-reset | phishing | 1.0000 | LABEL_3 at 1.0000 | 0.2132 | 0.9912 |

### URL branch, P(phishing) with the full URL, then with the scheme removed

Format: `with scheme / scheme stripped`. For candidate 9 the value is P(MALWARE).

| URL string | Expected | 5 urlbert-tiny-v4 | 7 ealvaradob bert-phishing-url | 8 pirocheto ONNX | 9 kmack |
|---|---|---|---|---|---|
| https://www.example.com/login | benign | 0.0041 / 0.3200 | 0.0006 / 0.0004 | 0.5176 / 0.9955 | 0.9325 / 0.8666 |
| https://www.example.org/docs/getting-started.html | benign | 0.0009 / 0.0021 | 0.0002 / 0.0002 | 0.0391 / 0.4563 | 0.0481 / 0.6658 |
| https://accounts.example.org/signin | benign | 0.0205 / 0.9816 | 0.0081 / 0.0054 | 0.7445 / 0.9687 | 0.9975 / 0.9623 |
| http://secure-paypa1-account-verify.example.net/update | phishing | 0.9999 / 0.9999 | 0.9998 / 0.9998 | 1.0000 / 1.0000 | 0.9997 / 0.9987 |
| http://192.0.2.10/bank/login.php?user=verify&session=8813 | phishing | 0.9979 / 0.9997 | 0.9998 / 0.9998 | 0.9999 / 1.0000 | 0.9994 / 0.9930 |
| http://example-bank-security-alert.example.com/confirm/account/password-reset | phishing | 0.7678 / 0.8962 | 0.9998 / 0.9996 | 0.9002 / 0.9754 | 0.9995 / 0.9993 |

urlbert-tiny-v4 token counts for the 6 URLs were 19, 35, 27, 37, 40 and 54 (URL lengths 29 to 79 characters). Its limit is 64 tokens, no unknown tokens were produced.

Candidate 6 (DomURLs_BERT): loading it as a sequence classifier reports `classifier.weight` and `classifier.bias` as missing, the checkpoint contains only `embeddings`, `encoder` and `pooler`, and `config.architectures` is `["BertModel"]`. Mean pooled embeddings have shape 768. Cosine similarity between the 6 toy URLs ranged from 0.8557 to 0.9561, benign and phishing alike. A scikit-learn logistic head fitted on those 6 vectors in 10.4 ms, which proves the plumbing only and is not an accuracy figure.

### Image branch, CLIP

(a) Zero-shot with two prompts: `a login page asking for a password` versus `an ordinary web page`.

| Image | Cosine (login, ordinary) | Softmax P(login prompt) |
|---|---|---|
| images/login_form.png | 0.3259, 0.2575 | 0.9989 |
| images/article_page.png | 0.2000, 0.2578 | 0.0031 |
| images/timetable_page.png | 0.1933, 0.2234 | 0.0471 |
| images/sms_parcel.png | 0.2417, 0.2500 | 0.3032 |

Zero-shot with five prompts (login page, card details page, text message asking to confirm card details, news article, table of information):

| Image | Top prompt | Softmax of top prompt |
|---|---|---|
| images/login_form.png | a screenshot of a login page asking for a password | 0.9824 |
| images/sms_parcel.png | a screenshot of a text message asking to confirm card details | 0.7717 |
| images/article_page.png | a screenshot of a news article or blog post | 0.8897 |
| images/timetable_page.png | a screenshot of a table of information | 0.9789 |

Encoding the two text prompts once took 24.4 ms, so prompts can be cached and the per image cost is the image encoder (100.6 ms median).

(b) Image embeddings: 512 dimensions, float32, L2 norm 1.0 after normalisation. Cosine similarity between images: login versus SMS 0.7094, login versus timetable 0.5055, login versus article 0.4335, SMS versus timetable 0.6297, SMS versus article 0.4816, article versus timetable 0.4176. No logistic head was fitted because there are no labelled screenshots yet.

## c. Recommendation

### Primary model per branch

| Branch | Primary | Why |
|---|---|---|
| Text | ealvaradob/bert-finetuned-phishing | Named labels in the config (benign, phishing), Apache-2.0, trained on SMS and email text which matches the message branch, 6 of 6 toy messages as expected with the widest margins, and the only text model that did not drift on the dentist reminder. Cost: 1.34 GB, 143.9 ms on short messages, 2.37 s at 512 tokens. |
| URL | CrabInHoney/urlbert-tiny-v4-phishing-classifier | Apache-2.0, 14.8 MB, 7.7 ms, 6 of 6 toy URLs as expected when the full URL with scheme is passed. A different architecture and tokenizer from the text model, which keeps the branches distinct. |
| Image | openai/clip-vit-base-patch32 | Works both ways. Use zero-shot with a cached prompt set now, and switch to embeddings plus a fitted logistic head once labelled screenshots exist. |

Conditions attached to the URL primary: always pass the full URL including scheme (removing the scheme moved `accounts.example.org/signin` from 0.0205 to 0.9816), and treat anything beyond 64 tokens as truncated. The model card also lists a newer `CrabInHoney/urlbert-tiny-v5`, which was not tested in this spike.

### Rejected candidates and the evidence

| Candidate | Reason for rejection |
|---|---|
| cybersectony/phishing-email-detection-distilbert_v2.4.1 | Four classes with no names in the config, and the model card code and the dataset card disagree on what the classes mean. Scored the benign dentist reminder 0.9839 as class 1. Put all 6 URL strings, benign and phishing, into LABEL_3 at 1.0000. Fast (22.7 ms) but not interpretable enough to fuse. |
| aamoshdahal/email-phishing-distilbert-finetuned | No licence stated, which is a problem for a delivered project. Correct on the 6 messages but with narrower margins (0.9116 on the parcel message, 0.0923 on the dentist reminder). Kept as the fallback if the primary is too slow, subject to the licence question. |
| ElSlay/BERT-Phishing-Email-Model | No licence stated. Twice the latency of the DistilBERT options (50.2 ms short, 772.9 ms at 512 tokens) with no visible gain on the samples. |
| ealvaradob/bert-phishing-url | Its own card says it is not the final model. 1.34 GB and 139.7 ms, about 18 times slower than urlbert-tiny-v4, and the same family and training data as the text primary, so it adds little diversity. Noted strength: outputs were stable when the scheme was removed. Keep as the fallback if urlbert-tiny-v4 proves too brittle in the real evaluation. |
| pirocheto/phishing-url-detection | Fastest (1.9 ms) and MIT, but weak on the benign samples: 0.5176 for `https://www.example.com/login`, 0.7445 for `https://accounts.example.org/signin`, and 0.9955 for `www.example.com/login` with the scheme removed. The repo's default file is a pickle, only the ONNX file was used. Useful as a cheap baseline to report against. |
| kmack/malicious-url-detection | Labels are BENIGN and MALWARE, not phishing. Flagged 2 of 3 benign samples as MALWARE with the scheme (0.9325, 0.9975) and 3 of 3 without it. |
| amahdaouy/DomURLs_BERT | NEEDS HEAD. Encoder only, no classifier weights, no licence stated, training data section of the card empty. Would need a labelled URL set and a fitted head before it produces any score, and raw embeddings of the toy URLs were all close together (cosine 0.86 to 0.96). |

### Evaluation leakage to avoid

- The text primary and the URL primary were both trained on `ealvaradob/phishing-dataset`. That dataset card lists its sources: a Kaggle phishing email set (Enron based), a Mendeley SMS set (5,971 messages), a Kaggle URL set (800,000 plus URLs, partly from JPCERT), and a Mendeley website set (PhishTank, OpenPhish, PhishRepo, Ebbu2017). Do not evaluate on that dataset or on those sources.
- Because both branches share one training source, their errors may be correlated. The fusion weights should be checked on data from outside it.
- If the fallbacks are used: aamoshdahal was trained on Enron, CEAS 2008, Ling-Spam, SpamAssassin, Nazario and Nigerian fraud emails. ElSlay on `zefang-liu/phishing-email-dataset`. pirocheto on `pirocheto/phishing-url`. kmack on `kmack/Phishing_urls`.
- CLIP was not trained on phishing data, so there is no direct leakage, but it is not validated for this task either.

## d. Things that could block or slow a full build tonight

Nothing is a hard blocker. All three primaries download, load and run on this machine. Items to handle:

1. Download failure on first attempt. 4 of 10 repos (ealvaradob/bert-finetuned-phishing, amahdaouy/DomURLs_BERT, ealvaradob/bert-phishing-url, openai/clip-vit-base-patch32) failed the first `snapshot_download` with `OSError(22, 'A required privilege is not held by the client')`, a Windows symlink privilege error in the Hugging Face cache. Large files had already arrived. Rerunning `02_download.py` completed all four in under 1 second each. The build should wrap downloads in a retry. Full download times for those four were therefore not measured.
2. Slow library import. Importing torch plus transformers model classes took 6.9 to 8.7 s per process because TensorFlow and JAX are installed and get imported. With `USE_TORCH=1`, `USE_TF=0`, `USE_FLAX=0` set, the same import took 3.7 to 3.8 s (3 runs each). Set these in the app before importing transformers.
3. Pickle weights. The text primary and CLIP ship only `pytorch_model.bin` on the main revision. transformers 4.48.0 loads it with `torch.load(weights_only=True)` (checked in the installed source). Safetensors conversions exist only on pull request refs (`refs/pr/4` for the text model, many for CLIP). Loading from those refs was not tested.
4. Worst case latency. Summing the measured medians gives about 2.5 s per item when the message reaches 512 tokens (2374.2 + 7.7 + 100.6 ms) and about 0.25 s for short messages. This is a sum of separate measurements, the three models were not timed together.
5. No labelled screenshots. The CLIP logistic head cannot be fitted until a labelled image set exists. Zero-shot works today but the two prompt version scored the SMS screenshot only 0.3032, so use a wider prompt set.
6. Licences. CLIP has no licence field in its card metadata, and the card says deployed use is out of scope. This needs a decision before delivery. It was not resolved in this spike.
7. Labels. urlbert-tiny-v4 has generic `LABEL_0` and `LABEL_1` in its config. The mapping (0 good, 1 phishing) comes from the model card and matched the sample behaviour. Hard code it in the app.
8. Pin revisions. Use the revision hashes in the table so results stay reproducible.

## Files

| File | Purpose |
|---|---|
| `01_metadata.py` | Hub metadata and model cards, writes `metadata.json` and `cards/` |
| `02_download.py` | Pinned downloads, writes `downloads.json` |
| `03_make_images.py` | Draws the 4 synthetic screenshots into `images/` |
| `04_bench.py` | Benchmarks one candidate in a fresh process, writes `results/<repo>.json` |
| `05_run_all.py` | Runs every candidate and merges into `results.json` |
| `06_extra_checks.py` | Import time, safetensors refs, torch.load check, dataset cards, writes `extra_checks.json` |
| `spike_inputs.py` | The harmless inputs |
| `downloads_first_attempt.json` | Record of the first download attempt including the 4 failures |
| `results.json` | All measurements |

To rerun: `python 01_metadata.py`, `python 02_download.py`, `python 03_make_images.py`, `python 05_run_all.py`, `python 06_extra_checks.py`.
