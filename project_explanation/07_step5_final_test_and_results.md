# 7. Step 5: the final test, and what the results mean

## 7.1 Freezing the setup

Before the test receipts were touched, every setting was fixed and nothing was changed after that:

| Setting | Value |
|---|---|
| Model | `models/Qwen3-VL-4B-Instruct-8bit` |
| Adapter | `models/lora_v1/epoch2` (chosen on dev100) |
| Prompt | `src/prompt_ft.txt` |
| Image limit | about 1 megapixel |
| Decoding | greedy, at most 2,048 tokens |
| Cleanup | on |

## 7.2 Running once

```bash
.venv/bin/python src/run_model.py --split test --run test_final --adapter models/lora_v1/epoch2 --prompt-file src/prompt_ft.txt
.venv/bin/python src/evaluate.py --split test --run test_final
```

The run covered 370 receipts in 2.2 hours, about 21 seconds each, with peak memory 6.4 GB. The scoring ran right after it. That was the **first time** the test labels were read.

## 7.3 The results

| Test split, 370 receipts | Score |
|---|---|
| **Overall accuracy** (headline) | **96.22%** |
| Exact accuracy | 92.47% |
| 95%-similar accuracy | 97.45% |
| Line-item F1 | 83.42% (precision 83.53%, recall 83.30%) |
| Receipts with every line item right | 75.95% |
| Receipts with all 8 fields right (strictest) | 44.32% |
| Valid JSON | 99.73% (369 of 370) |

| Field | Overall | Exact |
|---|---|---|
| company | 95.14% | 92.43% |
| date | 98.92% | 98.92% |
| address | 87.84% | 64.32% |
| tax_number | 94.59% | 94.59% |
| total | 98.65% | 98.65% |
| tax_amount | 98.92% | 98.92% |
| service_charge | 99.46% | 99.46% |

**Empty fields.** The model never invented a value where the label is empty: there were 0 hallucinations on every field. It left a value out only twice, once for a service charge and once for a whole item list.

## 7.4 How sure can we be? (the 95% range)

370 receipts is a sample. With a different set of similar receipts the score would come out a little different. How much different?

**Bootstrap** is a simple way to estimate this:
1. From the 370 test receipts, draw 370 at random **with replacement**, so some receipts appear twice and some not at all.
2. Compute the score on that new set.
3. Repeat 20,000 times.
4. Look at the middle 95% of the 20,000 scores.

| Measure | Score | 95% range |
|---|---|---|
| Overall accuracy | 96.22% | 95.29% to 97.07% |
| Exact accuracy | 92.47% | 91.39% to 93.47% |

Even the bottom of each range is above the 90% target. So the result is not luck from one particular set of receipts. The calculation used `results/test_final/per_receipt.csv`.

## 7.5 Looking at whole receipts

Under the overall rule, counting the 7 single-value fields:

| Receipts | How many fields wrong |
|---|---|
| 289 (78.1%) | none |
| 69 | 1 |
| 11 | 2 |
| 1 (receipt81) | all |

61.6% of the receipts have all 7 single-value fields right **and** every line item right.

## 7.6 The one broken answer: receipt81

receipt81's answer is not valid JSON. In one line item, the model forgot to write the amount and started the next item without closing the previous one:

```
... "unit_price": "15.80", {"description": "3/4\" ALUMINIUM ROD", ...
```

As the rules say, it counts as wrong on every field. It was **not** fixed after the test: changing anything after seeing test results would make the test unfair.

## 7.7 Dev100 compared with test

| | dev100 (100 train receipts, never taught) | test (370) |
|---|---|---|
| Overall | 97.29% | 96.22% |
| Exact | 94.14% | 92.47% |
| Line-item F1 | 84.98% | 83.42% |

The test is about 1 point lower. That is normal: dev100 was used to choose the version, so it is slightly favourable, while the test is completely new.

## 7.8 The whole journey in one table

| Step | Where measured | Overall | Exact | Line-item F1 |
|---|---|---|---|---|
| Model as it is (prompt v1) | dev100 | 91.00% | 87.57% | 31.00% |
| Better instructions + cleanup (prompt v2) | dev100 | 91.86% | 87.86% | 47.99% |
| Fine-tuned, epoch 1 | dev100 | 96.14% | 92.43% | 80.14% |
| Fine-tuned, epoch 2 (final) | dev100 | 97.29% | 94.14% | 84.98% |
| **Final model on the test** | **test** | **96.22%** | **92.47%** | **83.42%** |

## 7.9 What the numbers do NOT tell you

- **They hold for this kind of receipt only:** Malaysian shop receipts, mostly scanned, with Malaysian 12-digit GST numbers. Indian GST bills, phone photos, handwriting or other languages were never measured. Expect lower accuracy there, especially for the tax number and the line items. To know the real number, label 30 to 50 of those receipts and score them.
- **96% is per field, not per receipt.** 78% of receipts have every single-value field right. If receipts are to be processed with no human check, the per-receipt numbers matter more.
- **The headline rule is the forgiving one for spaces and punctuation.** The strict number is 92.47%, which is also above 90%.
- **The scores are measured against the corrected labels** (chapter 1.5).
