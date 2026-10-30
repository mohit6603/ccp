# Receipt Information Extraction with Qwen3-VL-4B

Structured-data extraction from receipt images using a vision-language model (VLM).
Input: a receipt image. Output: a JSON document with a fixed schema. Success is measured
by field-level accuracy against labelled data, and **accuracy is the top priority** of this project.

| | |
|---|---|
| Document status | Implementation plan v1.1, 2026-10-29. Phase 0 (setup, smoke test, label audit) is in progress; progress log in Section 11. |
| Model | Qwen3-VL-4B-Instruct |
| Compute | Google Colab T4 GPU for model inference and fine-tuning; local laptop for evaluation and analysis |

---

## 1. Objective

Given a photographed or scanned receipt, produce exactly this JSON structure:

```json
{
  "company": "INDAH GIFT & HOME DECO",
  "date": "19/10/2018",
  "address": "27, JALAN DEDAP 13, TAMAN JOHOR JAYA, 81100 JOHOR BAHRU, JOHOR.",
  "tax_number": null,
  "line_items": [
    {"description": "PASIR HALUS 1 LOAD (SMALL)", "quantity": 1, "unit_price": "148.40", "amount": "148.40"}
  ],
  "total": "60.30",
  "tax_amount": null,
  "service_charge": null
}
```

Rules for the output:
- Same keys, same order, same value types as the provided labels.
- If a field is not present on the receipt, its value is `null` (never an empty string, never a guess).
- `line_items` is `null` when no items can be read, otherwise a list of objects with
  `description`, `quantity`, `unit_price`, `amount`.

**Model:** Qwen3-VL-4B-Instruct (Hugging Face repository `Qwen/Qwen3-VL-4B-Instruct`). Third-party
listings and a repackaged copy of the weights state the license is Apache 2.0. The official model card
was not retrieved during planning, so the license field on that page is to be checked once before the
first download (Section 9).

---

## 2. Dataset

The dataset was prepared and split by the project owner. It is used as provided and is never modified.

| Item | Location | Count |
|---|---|---|
| Training images | `train/receipt_images/receiptN.jpg` | 600 |
| Training labels | `train/receipt_labeled_text/receiptN.json` | 600 |
| Test images | `test/receipt_images/receiptN.jpg` | 370 |
| Test labels | `test/receipt_labeled_text/receiptN.json` | 370 |
| Train/test assignment | `split_mapping.csv` | 970 rows |
| Traceability to original files | `image_mapping.csv` | 970 rows |
| Label provenance and review flags | `labeled_text_sources.csv` | 970 rows |

Image and label share the same base name (`receipt2.jpg` <-> `receipt2.json`). The same name is used
for every generated output, so any receipt can be traced across input, output and label.

The original filenames (for example `X00016469612.jpg`) and the `task1`/`task2` folder names in
`image_mapping.csv` indicate the receipts come from the public SROIE benchmark (ICDAR 2019 competition
on scanned receipt OCR and information extraction). The original SROIE set has 1000 receipts; this
970-receipt dataset was customised by the project owner to fit the requirements of this project
(selection of receipts, extended label schema, and the train/test split). Two points follow from public
information:
- SROIE's own license/terms of use could not be confirmed during planning, so its commercial-use status is
  **unverified**; the terms must be checked on the original dataset host before any use beyond internal
  evaluation (Section 9).
- Third-party sources report that the original SROIE annotations contain a few mistakes, which is one more
  reason for the label audit in Phase 0.

### 2.1 Label quality (important for interpreting accuracy)

Profiling of `labeled_text_sources.csv` shows the labels are **not all manually verified ground truth**:

| Property | Finding |
|---|---|
| `label_source = ground_truth` | 620 of 970 |
| `label_source = generated_from_ocr_text` | 350 of 970 |
| New fields (`tax_number`, `line_items`, `tax_amount`, `service_charge`) | All 970 generated from annotated OCR text |
| `needs_review = yes` | 41 of 970 (some `line_items` / `tax_amount` were dropped during labelling) |

Consequence: if a label is wrong, a correct model prediction is scored as an error. To control this we
add a **label audit step** (Phase 0) and report scores on label-quality slices (Section 6.4).

### 2.2 Observed value formats

- `date` is stored **as printed** on the receipt. 8+ patterns exist (`24/05/2017`, `25-04-18`, `11 MAY 2018`,
  `01/05/16`, `6/2/2017` ...). Scoring therefore needs a date normaliser (Section 6.2).
- Amounts are strings with 2 decimals (`"60.30"`); `unit_price` sometimes has 3 (`"9.000"`).
  Negative amounts exist (`"-9.99"`, for discounts/rounding).
- In line items, `quantity` is mostly an integer (rarely a float or `null`), `unit_price` may be `null`
  (about 29% of items), `amount` is always a string.
- Null rates in train: `service_charge` 94%, `line_items` 16%, `tax_number` 5.5%, `tax_amount` 5.5%.
  Because `service_charge` is almost always null, **null-handling accuracy matters as much as reading accuracy**.
- Image sizes vary widely, from about 443x959 to 4961x7016 pixels. Images are resized to a bounded
  resolution before inference (Section 5.2).

---

## 3. High-level flow

```
                      +------------------------------+
 train/receipt_images |                              |   outputs/<run>/receiptN.json
 (600 images)  ---->  |  Qwen3-VL-4B  +  prompt      | ---------------------------->  evaluate.py
                      |  (JSON schema, greedy decode)|                                   |
                      +------------------------------+                                   v
                                                         train/receipt_labeled_text/ --> results/<run>/
                                                                (labels)                  scores + error report
                                                                                              |
                          analyse errors -> improve (prompt / resolution / fine-tune) <-------+
                                                   |
                                                   v
              FINAL: test/receipt_images (370, labels hidden) -> model -> outputs/test_final/
                                                   |
                                                   v
              compare with test labels (only now) -> results/test_final/  (reported accuracy)
```

---

## 4. Project structure (GitHub repository)

```
VLMOCR_MODEL/  [on GitHub]
├── train/                      Provided dataset (600 receipt images + labels)
├── test/                       Provided dataset (370 receipt images + labels, used once at the end)
├── image_mapping.csv           Traceability to original files
├── labeled_text_sources.csv    Label provenance and review flags
├── split_mapping.csv           Train/test assignment
│
├── src/
│   ├── prompt_v1.txt           Baseline prompt (new prompts: prompt_v2.txt, ...)
│   ├── inference.py            Run model on a folder of images -> one JSON per image
│   ├── evaluate.py             Compare generated JSONs with labels -> scores (to be written)
│   ├── make_audit_sample.py    Selects the receipts for the label audit (fixed seed)
│   ├── make_colab_package.py   Rebuilds the Colab zips and notebook
│   └── (finetune script)       Colab notebook, added only if fine-tuning is needed
│
├── colab/
│   └── run_on_colab.ipynb      Google Colab T4 notebook for inference + fine-tuning
│
├── label_audit/                Phase 0 audit notes; corrected labels stored separately
├── README.md                   This file
├── requirements.txt            Python dependencies
└── .gitignore                  Excludes: .venv, models, outputs, results, *.zip
```

**Where is what (on GitHub):**
- Input data: `train/receipt_images/` and `test/receipt_images/`
- Labels (reference answers): `train/receipt_labeled_text/` and `test/receipt_labeled_text/`
- Audit progress: `label_audit/audit_notes.csv` and `audit_sample.csv`
- Code: `src/*.py`

Original labels are never overwritten. Any label correction is stored in `label_audit/`.

---

## 4.1 What's NOT on GitHub (and why)

The following are excluded from the GitHub repository because they are generated/local-only or too large:

| Folder/File | Size | Reason | How to recreate |
|---|---|---|---|
| `.venv/` | ~500MB | Python virtual environment (user-specific) | `python -m venv .venv && pip install -r requirements.txt` |
| `models/` | ~9GB | Model weights (Qwen3-VL-4B-Instruct) | Run `colab/run_on_colab.ipynb` Cell 1, or download manually from Hugging Face |
| `outputs/` | Variable | Generated model outputs (receipts JSONs after running inference) | Run inference on Colab (Phase 1) or locally via `src/inference.py` |
| `results/` | Variable | Evaluation metrics and error analysis | Run `src/evaluate.py` after generating outputs |
| `colab/*.zip` | 538MB | Pre-packaged training/test images for Colab upload | Auto-generated by `src/make_colab_package.py` |

**Clone & Setup:**
```bash
git clone https://github.com/mohit6603/ccp.git
cd ccp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Model weights download on first run of inference
```

---

## 5. Implementation plan

### Phase 0 - Setup, smoke test and label audit
1. Create a Python virtual environment; install PyTorch (CPU) and Hugging Face Transformers.
2. **Smoke test:** run the model on a few training images to measure seconds per image and memory use,
   and to check that the output is parseable JSON. Done on the laptop CPU with one image (result in
   Section 5.2: too slow for the full runs); repeated on a Colab T4 with 5 images before the baseline.
3. **Label audit:** visually compare labels with the receipt images for a sample, to estimate the true
   label error rate:
   - all 41 receipts with `needs_review = yes`
   - a random sample of about 40 receipts with `label_source = generated_from_ocr_text`
   - a random sample of about 20 receipts with `label_source = ground_truth`
   
   Disagreements are logged in `label_audit/`; suspected label errors are flagged for the project owner to
   confirm. The audit is sample-based (not all 970), because exhaustive visual checking is slow and
   itself error-prone. After the baseline, receipts where model and label disagree are the best
   candidates for a second review. The measured label error rate gives the practical accuracy ceiling.

### Phase 1 - Baseline on the 600 training images (Colab T4, zero-shot)
1. Prompt contains the exact JSON schema, the null rule, and the format rules from Section 1.
2. Deterministic decoding (no sampling) so that runs are reproducible.
3. For each image the response is parsed and saved as `outputs/train_baseline/receiptN.json`
   (same base name as the input). The raw text goes to `raw/`. If JSON parsing fails, the receipt is
   recorded as invalid (counted in the validity metric, not silently repaired).
4. The script is **resumable**: it skips receipts whose output already exists, so a run can be stopped
   (or interrupted by a Colab disconnect) and continued.
5. `run_config.json` records everything needed to reproduce the run.

### Phase 2 - Evaluation
`evaluate.py` compares every `outputs/<run>/receiptN.json` with its label and writes the results to
`results/<run>/`. Metrics are defined in Section 6.

### Phase 3 - Error analysis and improvement
Errors are grouped by field and by cause (misread digits, wrong field picked, hallucinated value where
the label is null, missed value, invalid JSON, label error). Improvements are tried in this order,
cheapest first, and each is a separate, named run compared against the baseline:
1. **Prompt:** clearer field definitions, explicit null rule, examples of tricky cases.
2. **Image resolution:** raise or lower the maximum image size (affects reading of small print vs. speed).
3. **Output handling:** strict JSON parsing with one controlled retry on invalid output.
4. **Fine-tuning (LoRA / QLoRA) on Google Colab T4** with the training images and labels, if the above
   does not reach the target. Notes:
   - T4 has 16 GB VRAM and no bf16 support, so training runs in fp16 with 4-bit quantised base weights.
   - To get an honest measure of generalisation, about 80 training receipts are held out as a
     development set (random, fixed seed) and are not used for training. The zero-shot baseline in
     Phase 1 still runs on all 600 images, because no training is involved there. The remaining
     ~520 receipts are used for fine-tuning.
   - The resulting adapter is downloaded and inference + evaluation continue locally.
   - Only public/synthetic data goes to Colab. No real customer or card data is ever uploaded.

Every change is judged on the same metrics; a change is kept only if it improves accuracy.

### Phase 4 - Final test (370 images)
1. The configuration is **frozen** (model, adapter, prompt, resolution) before touching the test set.
2. The model receives **only the images**. Test labels are not read by the inference script.
3. Outputs are written to `outputs/test_final/receiptN.json`.
4. Only after generation, `evaluate.py` compares them with `test/receipt_labeled_text/` and writes
   `results/test_final/`.
5. The test set is used once. It is never used to tune the prompt or the model.

### 5.1 Compute
| Stage | Where | Reason |
|---|---|---|
| Model inference (baseline runs, improved runs, final test) | Google Colab T4 GPU | Measured: the laptop CPU needs about 250 s per receipt (Section 5.2) |
| Fine-tuning (only if needed) | Google Colab T4 GPU | Training is not feasible on the laptop |
| Evaluation, error analysis, label audit, reporting | Local laptop | Lightweight; needs only the generated JSON and the labels |

Local machine: Intel Core i5-1334U (12 threads), 15 GB RAM (about 10 GB free), integrated Iris Xe GPU
(no CUDA), 361 GB free disk.

**Colab workflow:** `colab/run_on_colab.ipynb` runs the same `src/inference.py` on a T4. Only the code and
the receipt **images** are uploaded (`colab/VLMOCR_train.zip`; the test images are in a separate
`colab/VLMOCR_test.zip` that is uploaded only for the final run). Labels are never uploaded to Colab for
inference. Outputs are written to Google Drive, then copied into `outputs/<run>/` on the laptop for
evaluation. The run is resumable if Colab disconnects.

### 5.2 Measured local CPU speed (why inference moved to Colab)
Smoke test on the laptop (Qwen3-VL-4B-Instruct, bf16, one 448x992 receipt, 777 input tokens, 234 output
tokens): **251 s per receipt**, peak RAM about 8.5 GB. At this speed 600 receipts would take about 40 hours.
The CPU has no native bf16 support (AVX2 / AVX-VNNI only), bf16 matrix multiplication is about 4x slower than
fp32 in a micro-benchmark, and fp32 (about 16 GB of weights) does not fit in 15 GB of RAM. Quantising to
8-bit could be faster but was not evaluated, and would require an accuracy check because accuracy is the
priority.

Colab settings (to be confirmed by the first GPU smoke test): fp16 weights (T4 has no native bf16), bounded
image resolution (receipts are tall and narrow, so small print needs enough pixels), greedy decoding.

---

## 6. Evaluation methodology

### 6.1 Metrics
Each field is scored in two ways, as agreed:

1. **Exact match (after normalisation)** - strict.
2. **Fuzzy match** - for text fields, similarity of at least 0.95 counts as correct, and the mean
   similarity is also reported. For numeric fields, fuzzy equals exact.

| Level | Metric |
|---|---|
| Validity | % outputs that are valid JSON; % that follow the schema (keys, types) |
| Per field | Exact accuracy and fuzzy accuracy for `company, date, address, tax_number, total, tax_amount, service_charge` |
| Null handling | For each field: correct-null rate, **hallucination rate** (model gives a value, label is null), **miss rate** (model gives null, label has a value) |
| Line items | Item-level precision, recall and F1 (items matched by description similarity, then compared on quantity, unit price, amount); receipt-level exact match of the whole list |
| Receipt level | % receipts with all fields correct (strict); average number of wrong fields per receipt |
| Headline | Macro-average of per-field exact accuracy, plus the receipt-level figure |

`line_items` is also reported separately from the scalar fields, because it is the hardest field and
its labels are the least reliable.

### 6.2 Normalisation (applied identically to prediction and label)
- **Text** (`company`, `address`, `tax_number`, item `description`): trim, collapse repeated whitespace,
  case-insensitive. Punctuation is kept, so differences in punctuation are still reported as errors.
- **Amounts** (`total`, `tax_amount`, `service_charge`, `unit_price`, `amount`): compared as decimal
  numbers, so `"9.000"` equals `"9.00"`. Sign is preserved.
- **Date:** parsed to a calendar date (day-first, 2-digit years expanded, month names recognised) and
  compared as dates. The raw string comparison is reported as well.
- **Null:** `null` only matches `null`. An empty string is treated as a non-null value, hence an error.

Normalisation rules are defined once, before the first scoring run, and are not changed afterwards to
raise scores.

### 6.3 Output artefacts per run (`results/<run>/`)
- `summary.csv` - headline and per-field metrics
- `per_receipt.csv` - for each receipt and field: label, prediction, exact flag, fuzzy score
- `per_field.csv` - null-handling and field-level breakdown
- `errors.csv` - only the mismatches, for analysis

### 6.4 Label-quality slices
All metrics are reported for: (a) all receipts, (b) only `ground_truth` labels, (c) excluding
`needs_review = yes`. This separates model errors from label noise.

---

## 7. Data handling and compliance

- The dataset consists of receipt images and derived JSON; it is treated as public benchmark data and used
  for internal evaluation only until its terms of use are verified (Section 9).
- No real card numbers (PAN), CVV, expiry dates or personal identifiers are to be processed. If any
  real sensitive data is found in an image or label, work stops and the data is sanitised first.
- Any ISG or customer data used later must be synthetic or sanitised before being used with the model
  or uploaded to Colab.
- Model weights are downloaded from the official Qwen organisation page only. Files in this project
  are stored locally.

---

## 8. Reproducibility

- Greedy decoding, fixed random seed for any sampling-dependent step (train/dev split, audit sample).
- Each run stores `run_config.json`: model identifier and version, adapter (if any), prompt text, image
  resolution, decoding settings, library versions, date and time.
- Run names are explicit (`train_baseline`, `train_prompt_v2`, `train_lora_v1`, `test_final`).

---

## 9. Assumptions, risks and open items

**Decisions agreed with project owner**
- A missing field is stored as `null`.
- Both exact (normalised) and fuzzy matching are reported.
- Colab T4 is used for model inference and fine-tuning; the laptop for evaluation. (Originally local CPU
  inference was chosen; changed after the measured 251 s per receipt, Section 5.2.)
- The 41 `needs_review = yes` receipts are scored and also reported separately (Section 6.4).
- About 80 training receipts are held out as a development set if fine-tuning happens (Section 5, Phase 3).
- Label audit is sample-based (about 100 receipts), with follow-up on model/label disagreements.

**Open items (to be checked before or during Phase 0)**
- ~~Qwen3-VL-4B-Instruct license~~ - closed: the official Hugging Face model card lists `apache-2.0`
  (checked 2026-10-29; the full LICENSE text was not read).
- SROIE dataset terms of use: could not be confirmed; check on the original dataset host. Until then the
  data is used for internal evaluation only.
- Whether Colab upload of the receipts is acceptable under company policy (the receipts are public
  benchmark data, but this is to be confirmed by the project owner).

**Risks**
| Risk | Mitigation |
|---|---|
| Labels generated from OCR text may be wrong, which lowers measured accuracy | Label audit; slice reporting; flagged disagreements reviewed |
| `line_items` labels dropped when unreliable, so `null` may hide real items | Report `line_items` separately; check in audit |
| Laptop CPU inference is far too slow (measured 251 s per receipt) | Inference runs on Colab T4; evaluation stays local |
| Colab free-tier limits (GPU quota, session disconnects) | Resumable runs with outputs saved to Google Drive; GPU smoke test first |
| fp16 on T4 could behave differently from the CPU bf16 run | 5-image GPU smoke test checks output validity before the full run |
| Lower precision (8-bit) could reduce accuracy | Compare against 16-bit on a sample before adopting |
| Fine-tuning on 600 samples may overfit | Held-out dev set; test set used once |
| Zero-shot accuracy of a 4B model on receipts is unknown; public evidence is thin | Baseline measured first; improvements decided by data |

---

## 10. Deliverables

1. Generated JSON for every training image (`outputs/train_baseline/` and later runs).
2. Evaluation reports for every run (`results/<run>/`).
3. Error analysis and the list of improvements tried, with their measured effect.
4. Final JSON outputs for the 370 test images (`outputs/test_final/`) and the final accuracy report
   (`results/test_final/`).
5. This README, kept up to date with the final configuration and results.

---

## 11. Progress log (as of 2026-10-29)

**Done**
- Python environment, model download (Qwen3-VL-4B-Instruct, both weight files verified against the
  official SHA-256), `src/inference.py`, baseline prompt `src/prompt_v1.txt`.
- Colab package built (`colab/`): notebook plus code/image zips, no labels inside.
- Smoke test on the laptop CPU (1 receipt): output was valid JSON, but 251 s per receipt, so inference
  moved to Colab T4 (Section 5.2).
- Label audit started: 9 of the 101 selected receipts reviewed (`label_audit/audit_notes.csv`).

**Early findings (not yet statistically meaningful)**
- Label audit, 9 receipts, all from the `needs_review = yes` group: 6 labels match the image, 2 match with
  small caveats (whitespace differences, clipped or degraded print), 1 has a real error (`line_items` is
  `null` although the receipt prints items).
- First model output (train receipt2): company, date and total correct; the items were read correctly while
  the label has `line_items = null` (a label error); the model wrongly returned a member number as
  `tax_number` and a rounding adjustment (-0.01) as `tax_amount` (model errors; the labels `null` are right).
  Address matched the print exactly (`27,JALAN`), whereas the label has a space after the comma, so exact
  matching needs whitespace-around-punctuation normalisation and fuzzy matching covers the rest.
- Of the 9 audited receipts, 4 belong to the test split. They were looked at only to check labels. Any
  prompt rule derived from observations is validated on **train** errors only, so the test set stays clean
  for the final evaluation.

**Next**
1. Run the 5-image GPU smoke test in `colab/run_on_colab.ipynb` (Cell 4) and check speed and validity.
2. Run the baseline on all 600 training images on Colab; copy `outputs/train_baseline/` to the laptop.
3. Write `src/evaluate.py` (Section 6) and score the baseline.
4. Continue the label audit on receipts where model and label disagree.
5. Error analysis, improvements (Section 5, Phase 3), then the single final test run.

---

## 12. Continuing on another machine (for example a MacBook)

Copy the project folder **without** `.venv/` (laptop-specific) and `models/` (about 9 GB of weights, only needed
for local model runs).
1. Install Python 3.10 or newer, create a virtual environment, then `pip install -r requirements.txt`
   (enough for evaluation, label audit and reporting).
2. Model inference runs on Colab: upload `colab/VLMOCR_train.zip` to Google Drive folder `MyDrive/VLMOCR/`
   and open `colab/run_on_colab.ipynb` (Section 5.1). `python src/make_colab_package.py` rebuilds the zips
   if the images or code change.
3. Copy the Colab outputs from Drive into `outputs/<run>/` in this project, then evaluate locally.
