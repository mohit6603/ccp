# Receipt information extraction with Qwen3-VL-4B

The model reads a receipt image and returns one JSON object with 8 fields, in the same format as the label files:
- company
- date
- address
- tax_number
- line_items
- total
- tax_amount
- service_charge

Everything runs locally on a MacBook Air (Apple M5, 16 GB) with MLX. No receipt is sent to an online model or API.

## Result

The fine-tuned model scores **96.22% overall accuracy** on the 370 test receipts, above the 90% target. The model and its settings were chosen using train receipts only (dev100). The test split was then run once.

| Test split, 370 receipts | Score |
|---|---|
| **Overall accuracy** (headline, see [Scoring rules](#scoring-rules)) | **96.22%** |
| Exact accuracy | 92.47% |
| 95%-similar accuracy | 97.45% |
| Line-item F1 | 83.42% (precision 83.53%, recall 83.30%) |
| Receipts with every line item right | 75.95% |
| Receipts with all 8 fields right (strict) | 44.32% |
| Valid JSON output | 99.73% (369 of 370) |

| Field | Overall | Exact |
|---|---|---|
| company | 95.14% | 92.43% |
| date | 98.92% | 98.92% |
| address | 87.84% | 64.32% |
| tax_number | 94.59% | 94.59% |
| total | 98.65% | 98.65% |
| tax_amount | 98.92% | 98.92% |
| service_charge | 99.46% | 99.46% |

- **Comparison on dev100.** dev100 is 100 train receipts the model never trained on. Fine-tuning raised overall accuracy there from 91.86% to 97.29%; 91.86% was the best result without fine-tuning. Line-item F1 rose from 47.99% to 84.98%. Every run is listed in [All runs](#all-runs).
- **Null fields on test.**
  - The model never filled a field that the label leaves empty.
  - It left a field empty that the label has only twice: one service charge, and one line-item list.
- **How certain the score is.** Resampling the 370 test receipts gives a 95% range of 95.29–97.07% for overall accuracy, and 91.39–93.47% for exact accuracy. Both ranges are above the 90% target. They were computed from `results/test_final/per_receipt.csv` (20,000 bootstrap resamples of receipts).
- **Per receipt, under the overall rule:**
  - 289 of 370 test receipts (78.1%) have all 7 single-value fields right.
  - 69 have one field wrong, and 11 have two wrong.
  - 1 (receipt81, the invalid output) is wrong on every field.
  - 61.6% of receipts also have every line item right.
- **What the score covers.** It is measured on receipts from the same collection as the training receipts: SROIE, Malaysian shop receipts, mostly scanned. Other receipt types are not covered; see [Known limitations](#known-limitations).

## Where everything is

```
Clean Dataset/                       (on GitHub: mohit6603/ccp)
├── README.md
├── requirements.txt                 exact Python packages used (Apple Silicon Mac)
├── train/receipt_images/            600 input images (receiptN.jpg)
├── train/receipt_labeled_text/      600 labels (receiptN.json)
├── test/receipt_images/             370 input images, run once at the end
├── test/receipt_labeled_text/       370 labels, read only by the final scoring
├── metadata/                        label-fix log and rules, mapping to the original SROIE files
├── src/
│   ├── prompt_v1.txt                baseline prompt (field definitions)
│   ├── prompt_v2.txt                improved prompt (rules learned from the train errors)
│   ├── prompt_ft.txt                one-sentence instruction used with the fine-tuned model
│   ├── dev100.txt                   the 100 dev receipts (every 6th train receipt)
│   ├── run_model.py                 runs the model on a split -> one JSON per image
│   ├── evaluate.py                  compares outputs with labels -> scores
│   ├── make_finetune_data.py        builds the fine-tuning data from the train split
│   └── finetune.py                  LoRA fine-tuning on this Mac
├── finetune/                        fine-tuning data: train.jsonl (498), valid.jsonl (99), images/ (not in git; rebuilt by make_finetune_data.py)
├── models/
│   ├── Qwen3-VL-4B-Instruct/        official weights, 8.3 GB (not in git, see Setup)
│   ├── Qwen3-VL-4B-Instruct-8bit/   8-bit MLX conversion used by every run, 4.8 GB (not in git, see Setup)
│   ├── lora_v1/                     fine-tuning log and settings; the adapters epoch1/ and epoch2/ are not in git (see Setup)
│   └── download_retry.sh            download script for the official weights
├── outputs/<run>/receiptN.json      model output, same name as the input image
├── outputs/<run>/raw/receiptN.txt   raw model text for every image
├── outputs/<run>/timing.csv         seconds, tokens and memory per image
├── outputs/<run>/run_config.json    model, adapter, prompt and settings of the run
└── results/<run>/                   summary.csv, per_field.csv, per_receipt.csv, errors.csv, line_items.csv
```

The runs in `outputs/` and `results/` are:
- train_baseline
- dev100_prompt_v2
- dev100_lora_v1_ep1
- dev100_lora_v1_ep2
- test_final

Any receipt can be traced end to end by its name. For example, receipt1 of the final test run:

| Stage | Location |
|---|---|
| Input image | `test/receipt_images/receipt1.jpg` |
| Model output | `outputs/test_final/receipt1.json` (raw text in `outputs/test_final/raw/receipt1.txt`) |
| Label | `test/receipt_labeled_text/receipt1.json` |
| Score | `results/test_final/per_receipt.csv`, the rows for receipt1 |
| Mistakes only | `results/test_final/errors.csv` |

## Setup

These steps are needed once on a new machine.

**What you need:**
- An Apple Silicon Mac (M1 or newer). MLX runs only on Apple Silicon. On other computers only the scoring runs; see [Replicating on another machine](#replicating-on-another-machine).
- 16 GB of memory for fine-tuning. Running the model needs about 7 GB.
- About 20 GB of free disk space.

Versions used:
- Python 3.13.13
- mlx 0.32.4
- mlx-vlm 0.7.11 (with transformers 5.22.1)
- rapidfuzz 3.14.6
- huggingface_hub 2.4.0

`requirements.txt` pins every package at the exact version used.

```bash
git clone https://github.com/mohit6603/ccp.git
cd ccp
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt

# official weights (bash models/download_retry.sh does the same and retries until both files are complete)
.venv/bin/hf download Qwen/Qwen3-VL-4B-Instruct --local-dir models/Qwen3-VL-4B-Instruct
shasum -a 256 models/Qwen3-VL-4B-Instruct/model-0000*-of-00002.safetensors

# 8-bit MLX conversion (group size 64)
.venv/bin/mlx_vlm.convert --hf-path models/Qwen3-VL-4B-Instruct --mlx-path models/Qwen3-VL-4B-Instruct-8bit -q --q-bits 8
shasum -a 256 models/Qwen3-VL-4B-Instruct-8bit/model.safetensors
```

Expected checksums. They were checked after the download and again on 2026-11-01. If the 8-bit checksum differs, the conversion is not identical and some outputs may differ slightly.

| File | sha256 |
|---|---|
| `Qwen3-VL-4B-Instruct/model-00001-of-00002.safetensors` | `30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9` |
| `Qwen3-VL-4B-Instruct/model-00002-of-00002.safetensors` | `046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6` |
| `Qwen3-VL-4B-Instruct-8bit/model.safetensors` (the conversion) | `b3d9193e10e795bc09572ddbf779c3cdb9e1d7e6c42556f5e15d0d048bf4e55f` |

**The fine-tuned adapter.** The adapter folders `models/lora_v1/epoch1/` and `epoch2/` are not in git. Each holds a 132 MB file, and GitHub refuses files over 100 MB. There are two ways to get them:
- copy the two folders from the original Mac into `models/lora_v1/`; or
- retrain them with the fine-tuning commands under [How to run](#how-to-run), which takes about 7 hours on an M5 with 16 GB.

A retrained adapter can differ slightly from the original.

## How to run

Run every command from this folder.
- Each run writes to `outputs/<run>/` and `results/<run>/`, under its own name.
- If a run is interrupted, rerunning the same command resumes it.
- If any setting differs from the run's saved `run_config.json`, the script stops and asks for a new run name.
- Long runs were started as `caffeinate -s -i <command>`, so that the Mac stays awake. Keep it on power with the lid open.

The final model on the test split, which is what produced the result above:

```bash
.venv/bin/python src/run_model.py --split test --run test_final --adapter models/lora_v1/epoch2 --prompt-file src/prompt_ft.txt
.venv/bin/python src/evaluate.py --split test --run test_final
```

New receipts, with no labels needed. Put the images in any folder. JPG, PNG, WEBP, BMP and TIFF are read; convert iPhone HEIC photos to JPG first.

```bash
.venv/bin/python src/run_model.py --images-dir ~/Desktop/new_receipts --run new_receipts --adapter models/lora_v1/epoch2 --prompt-file src/prompt_ft.txt
```

- The JSON for each image is written to `outputs/new_receipts/<image name>.json`, and the raw text to `outputs/new_receipts/raw/`.
- Without labels there is nothing to score, so `evaluate.py` is not used.
- This was tested on 2026-11-01 with a renamed JPG and a PNG in a folder outside the project. Both outputs were identical to the earlier runs on the same two receipts, at about 14 s per image.

Fine-tuning, then scoring each epoch on dev100. The scoring is shown for epoch 2; epoch 1 is the same with `epoch1` in place of `epoch2`.

```bash
.venv/bin/python src/make_finetune_data.py --prompt-file src/prompt_ft.txt    # writes finetune/
.venv/bin/python src/finetune.py --out models/lora_v1 --epochs 2               # about 7 hours
.venv/bin/python src/run_model.py --split train --run dev100_lora_v1_ep2 --names-file src/dev100.txt --adapter models/lora_v1/epoch2 --prompt-file src/prompt_ft.txt
.venv/bin/python src/evaluate.py --split train --run dev100_lora_v1_ep2 --names-file src/dev100.txt
```

The exact options of every run:

| Run | `run_model.py` options | `evaluate.py` options |
|---|---|---|
| train_baseline | `--split train --no-clean` | `--split train` |
| dev100_prompt_v2 | `--split train --names-file src/dev100.txt --prompt-file src/prompt_v2.txt` | `--split train --names-file src/dev100.txt` |
| dev100_lora_v1_ep1 | `--split train --names-file src/dev100.txt --adapter models/lora_v1/epoch1 --prompt-file src/prompt_ft.txt` | `--split train --names-file src/dev100.txt` |
| dev100_lora_v1_ep2 | `--split train --names-file src/dev100.txt --adapter models/lora_v1/epoch2 --prompt-file src/prompt_ft.txt` | `--split train --names-file src/dev100.txt` |
| test_final | `--split test --adapter models/lora_v1/epoch2 --prompt-file src/prompt_ft.txt` | `--split test` |

Notes on these options:
- **train_baseline** was run before the cleanup step existed. `--no-clean` reproduces how it was parsed.
- **`--names-file`** in `evaluate.py` writes to `results/<run>/` unless `--res-dir` is given. To score train_baseline on dev100 only, add `--res-dir` with a separate folder, so that `results/train_baseline` keeps the full 600-receipt scores.
- **Quick trials:** `run_model.py` accepts `--limit N` or `--names receipt2 receipt7`, and `evaluate.py --partial` scores only the receipts a run attempted.

## Replicating on another machine

**On an Apple Silicon Mac**, after [Setup](#setup), run the final model on the test split under a new run name and score it:

```bash
.venv/bin/python src/run_model.py --split test --run test_final_check --adapter models/lora_v1/epoch2 --prompt-file src/prompt_ft.txt
.venv/bin/python src/evaluate.py --split test --run test_final_check
```

- Compare `results/test_final_check/summary.csv` with `results/test_final/summary.csv`.
- With the same checksums and package versions, the outputs should be the same or very nearly the same. A different Apple chip or macOS version can change a few outputs slightly, through floating-point differences.
- Use a new run name. `outputs/test_final` is already complete in the repo, so rerunning that name runs nothing.

**On any other computer** (Windows, Linux or an Intel Mac), the scores can be recomputed from the saved outputs. Only rapidfuzz is needed:

```bash
python -m venv .venv
.venv/bin/pip install rapidfuzz==3.14.6        # Windows: .venv\Scripts\pip install rapidfuzz==3.14.6
.venv/bin/python src/evaluate.py --split test --run test_final --res-dir results_check/test_final
```

This gives the same numbers as `results/test_final/`, without running the model.

Running the model itself on such a computer needs a different runtime, because this code uses MLX, which works only on Apple Silicon. An earlier version of this project measured 251 s per receipt with PyTorch on an Intel Core i5 CPU. That is about 26 hours for the test split.

## Model and pipeline

- **Weights:** Qwen3-VL-4B-Instruct, from the official Qwen organisation page on Hugging Face (Apache 2.0 license). Every run uses the 8-bit MLX conversion.
- **Images:**
  - rotated upright from the EXIF data and converted to RGB;
  - scaled down to at most about 1 megapixel, with sides that are multiples of 32, which gives at most 1,024 image tokens;
  - never scaled up.
- **Decoding:** greedy (temperature 0) with at most 2,048 output tokens, so runs are repeatable.
- **Prompt:** the fine-tuned model gets the one-sentence instruction in `src/prompt_ft.txt`. It learned the label conventions from the training labels. The runs without fine-tuning used the longer rule-based prompts `prompt_v1.txt` and `prompt_v2.txt`.
- **Output check:**
  - The answer must be JSON with exactly the 8 keys, and every line item must have exactly description, quantity, unit_price and amount.
  - An invalid answer gets no `receiptN.json`. Its raw text is always kept in `raw/`.
- **Cleanup:** from prompt_v2 on, valid answers go through a formatting-only cleanup (`clean_values` in `run_model.py`). It removes:
  - currency symbols and tax letters from amounts ("RM6.69" becomes "6.69");
  - label text before the GST number ("GST REG #");
  - registration numbers in brackets after the company name;
  - "@ price" or "##" after item names.

  A key the model leaves out becomes null, and the string "null" becomes null. No train label contains any of the removed text. train_baseline was scored without cleanup.
- **Adapter check:** mlx-vlm skips adapter weights whose names don't match, without any error. So `run_model.py` checks that all 504 adapter tensors are in the model before the first image, and stops if not.
- **Speed:** about 21 s per receipt for the test run, 2.2 hours for 370 receipts, with peak memory 6.4 GB.

## Fine-tuning

| Setting | Value |
|---|---|
| Method | LoRA on the language model. The vision encoder and the 8-bit base weights stay frozen. |
| Layers | The q, k, v, o, gate, up and down projections of all 36 language-model layers (252 layers) |
| Rank, alpha, dropout | 16, 32 (scale 2.0), 0 |
| Trainable parameters | 33.03 million (0.74% of the model) |
| Training data | 498 train receipts: the train split minus dev100, and minus receipt443, receipt729 and receipt796, which have near-duplicates in test |
| Target | The receipt's label JSON (compact, same key order). Loss is computed on the answer tokens only. |
| Length | 2 epochs, 250 optimizer steps; batch 1 with gradient accumulation 4 |
| Optimizer | AdamW, no weight decay, seed 0. Learning rate 1e-4 after 10 warm-up steps, then cosine decay to 1e-5. |
| Memory and time | Gradient checkpointing. Peak 17.2 GB, so macOS used swap on the 16 GB Mac. About 97 s per optimizer step (4 receipts), 7 hours in total. |
| Validation loss (dev100) | 0.0224 after epoch 1, 0.0181 after epoch 2 |
| Choice | Epoch 2, which had the higher dev100 overall accuracy (97.29% vs 96.14%) and line-item F1 (84.98% vs 80.14%) |

What `models/lora_v1/` holds:
- `epoch1/` and `epoch2/`: the adapter after each epoch.
- `adapters.safetensors`: the final adapter, identical to `epoch2/`.
- `last/`: a safety copy written every 25 steps.
- `train_log.csv`: the loss per optimizer step and the validation loss per epoch.
- `train_config.json`: the training settings.

Only `train_log.csv` and `train_config.json` are in git; the adapter files are over GitHub's size limit (see [Setup](#setup)).

What `finetune/valid.jsonl` holds: the dev100 receipts except receipt729, a test near-duplicate that is kept out of every fine-tuning file. That leaves 99.

## Scoring rules

These rules were fixed before the first run and are applied the same way to the model output and the label. The overall accuracy rule was added on 2026-10-31, when the target was agreed with the manager; no other rule changed.

- **Overall accuracy (the headline; target above 90%):** the average over the 7 single-value fields.
  - Company, address and tax_number also count as correct when their letters and digits match after all spaces and punctuation are removed. So "27,JALAN" equals "27, JALAN.", but every letter and digit must still be right.
  - Date and amounts use the rules below.
- **Exact accuracy:** the same 7-field average, with punctuation counted.
  - For text fields (company, address, tax_number), extra spaces and letter case are ignored.
- **95%-similar accuracy:** the same average, but text fields count as correct when they are at least 95% similar (rapidfuzz ratio). This also forgives one or two misread letters in long text, so it is reported only as a secondary number.
- **Amounts:** compared as decimal numbers, so "9.000" equals "9.00". Thousands commas are ignored and the sign is kept.
- **Dates:** compared as calendar dates, so "11/05/2018" equals "11 MAY 2018". Dates are read day first; 2-digit years are expanded and month names are recognised.
- **Missing values:**
  - Null only matches null, and an empty string counts as a value.
  - A model value where the label is null counts as a hallucination.
  - A null where the label has a value counts as a miss.
- **Invalid output:** a receipt whose output is not valid JSON counts as wrong on every field.
- **Line items:**
  - Each label item is paired with at most one predicted item: the most similar description, with the amount as tie-break.
  - An item is correct when its description is at least 95% similar and its quantity, unit price and amount are all equal.
  - Precision is correct items over predicted items. Recall is correct items over label items.
  - A receipt's line items are right only when every item is correct and there are no extra items.
- **All 8 fields right:** the 7 single-value fields under the exact rule, plus fully correct line items.

## Data handling

- **Local only:** all processing ran on one Apple Silicon Mac, and no receipt was sent to an online model or API. The only download was the model weights, from the official Qwen organisation page on Hugging Face (checksums under [Setup](#setup)).
- **Test labels:** first read by the scoring of test_final, after the model was frozen. The prompt, the cleanup and the epoch were all chosen on train receipts or dev100, and nothing was changed after the test run. Re-scoring the saved outputs, for example to replicate the result, changes nothing.
- **Training data:** fine-tuning never saw dev100 or the three train receipts that have near-duplicates in test.
- **Dataset terms:** the receipts come from the public SROIE benchmark (ICDAR 2019). Its terms of use have not been confirmed, so check them before using the data or the fine-tuned adapter beyond internal evaluation.

## Dataset notes

- **Labels:** the labels in `train/` and `test/` are the corrected ones. All 970 were checked against their images on 2026-10-30, and 437 receipts were corrected (279 train, 158 test).
  - Every change, with the old value, the new value and the reason, is in `metadata/label_fix_log.csv`.
  - The rules used are in `metadata/label_fix_summary.md`.
  - The original labels are in this repository's history, at commit `876dabf`.
- **Mapping files** in `metadata/`:
  - `image_mapping.csv`: the original SROIE file name of each receipt.
  - `split_mapping.csv`: the train/test assignment.
  - `labeled_text_sources.csv`: where each original label came from.
- **Duplicates:** three train/test pairs are near-duplicates.
  - receipt443 and receipt442 are the same receipt.
  - receipt796 and receipt249 are the same receipt.
  - receipt729 and receipt746 are consecutive receipts with identical content.

  All of them stay in the dataset. The three train receipts are left out of the fine-tuning data, so the test score is not inflated.
- **Address spacing:** address labels are not consistent about the space after a comma; both "27,JALAN" and "27, JALAN" occur. Most exact-rule address mismatches come from this kind of spacing or punctuation difference, which is why the headline rule ignores spaces and punctuation.
- **Known label limitations:**
  - receipt531 and receipt948 have blanked item names.
  - receipt663 has a faded date.
  - receipt580 has an unclear digit in its GST number.

## All runs

**Overall** is the headline rule, **Exact** counts punctuation, and **95% similar** forgives small misreads in long text (see [Scoring rules](#scoring-rules)). All three are in each run's `results/<run>/summary.csv`.

| Run | Receipts | Setup | Valid JSON | Overall | Exact | 95% similar | Line-item F1 | Notes |
|---|---|---|---|---|---|---|---|---|
| train_baseline | train, 600 | prompt v1, no cleanup | 99.83% | **90.02%** | 86.64% | 92.07% | 36.11% | 3.7 h, 22 s per receipt, peak memory 6.1 GB |
| train_baseline, dev100 only | train, 100 | prompt v1, no cleanup | 100% | **91.00%** | 87.57% | 93.43% | 31.00% | the train_baseline outputs scored on the dev100 receipts only, for comparison; no results folder of its own (see the `--res-dir` note under [How to run](#how-to-run)) |
| dev100_prompt_v2 | train, 100 | prompt v2, cleanup | 100% | **91.86%** | 87.86% | 94.43% | 47.99% | best result without fine-tuning |
| dev100_lora_v1_ep1 | train, 100 | fine-tuned (epoch 1), cleanup | 100% | **96.14%** | 92.43% | 97.14% | 80.14% | |
| dev100_lora_v1_ep2 | train, 100 | fine-tuned (epoch 2), cleanup | 100% | **97.29%** | 94.14% | 98.14% | 84.98% | best on every measure; chosen as the final model; 21 s per receipt |
| **test_final** | **test, 370** | fine-tuned (epoch 2), cleanup | 99.73% | **96.22%** | 92.47% | 97.45% | 83.42% | final result, run once; 2.2 h, peak memory 6.4 GB |

The dev100 per-field overall accuracy, before and after fine-tuning:

| Field | prompt v2, no fine-tuning | fine-tuned (epoch 2) |
|---|---|---|
| company | 86% | 98% |
| date | 97% | 98% |
| address | 83% | 94% |
| tax_number | 87% | 95% |
| total | 93% | 99% |
| tax_amount | 97% | 98% |
| service_charge | 100% | 99% |

## Known limitations

- **Receipt type:** the test receipts come from the same collection as the training receipts: SROIE, Malaysian shop receipts, mostly scanned, with Malaysian 12-digit GST numbers. Accuracy on other kinds of receipt has not been measured. That includes:
  - other countries and tax-number formats;
  - phone photos;
  - handwriting;
  - other languages.

  The model will still return JSON for such receipts, but expect lower accuracy, especially for tax_number and line items. Before quoting a number for a new kind of receipt, label 30 to 50 of them and score them.
- **One invalid output:** in test_final, receipt81 is not valid JSON. The model left out one line item's amount and started the next item without closing the previous one. As the scoring rules say, it counts as wrong on every field. Nothing was changed after the test run.
- **Address is the weakest field:** 87.84% overall and 64.32% exact on test.
- **Line items:** 75.95% of test receipts have every line item right.
- **Test vs dev100:** the test score is about 1 point below dev100 (96.22% vs 97.29%), which is expected for new receipts.
