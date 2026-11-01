# Project explanation: start here

This folder explains the whole project from the beginning, in simple words. You do not need any machine-learning background.

Read the files in order. Each chapter covers one step of the pipeline. Every technical word is explained the first time it appears, and again in the [glossary](09_glossary.md).

| # | File | What you will learn |
|---|---|---|
| 1 | [01_problem_and_data.md](01_problem_and_data.md) | What the model must do, the receipts, the labels, and why the data is split into groups |
| 2 | [02_how_the_model_works.md](02_how_the_model_works.md) | What Qwen3-VL-4B is, how it "reads" an image, and why we use MLX and 8-bit |
| 3 | [03_step1_running_the_model.md](03_step1_running_the_model.md) | `run_model.py`: from an image to a JSON file, step by step |
| 4 | [04_step2_scoring.md](04_step2_scoring.md) | `evaluate.py`: how every answer is checked against its label |
| 5 | [05_step3_better_prompts.md](05_step3_better_prompts.md) | The first results, the mistakes, and how better instructions helped |
| 6 | [06_step4_fine_tuning.md](06_step4_fine_tuning.md) | Teaching the model with examples (LoRA), and choosing the best version |
| 7 | [07_step5_final_test_and_results.md](07_step5_final_test_and_results.md) | The one-time test, the final numbers, and what they mean |
| 8 | [08_replicate_and_use.md](08_replicate_and_use.md) | Setting it up on another computer (Mac or Ubuntu), new receipts, fixing problems |
| 9 | [09_glossary.md](09_glossary.md) | Every technical word, explained simply |
| 10 | [10_questions_your_manager_may_ask.md](10_questions_your_manager_may_ask.md) | Short answers to likely questions |

## The project in one minute

- **Input:** a scanned or photographed shop receipt.
- **Output:** one JSON file with 8 fields: company, date, address, tax number, the list of items bought, total, tax amount and service charge.
- **Model:** Qwen3-VL-4B, a free AI model that can look at an image and write text about it.
- **Machine:** a MacBook Air (Apple M5 chip, 16 GB memory). Nothing was sent to any online service.
- **How we got there:**
  1. We ran the model as it is, which scored 90.02% on the training receipts.
  2. We gave it better instructions and added a formatting cleanup.
  3. We taught it with 498 example receipts. This teaching step is called fine-tuning.
- **Final result:** on 370 receipts it had never seen, **96.22% of the fields are correct**. The target was above 90%.

## The pipeline at a glance

```
 (1) DATA   train/ = 600 receipts + correct answers (labels)
            test/  = 370 receipts + labels, kept aside for the very end
                 │
                 ▼
 (2) MODEL  Qwen3-VL-4B, made smaller (8-bit), running on the Mac with MLX
                 │
                 ▼
 (3) RUN    src/run_model.py:  image ──► model ──► JSON  ──► outputs/<run>/receiptN.json
                 │
                 ▼
 (4) SCORE  src/evaluate.py:   compare each JSON with its label ──► results/<run>/
                 │
                 ▼
 (5) IMPROVE  better instructions (prompt v2) + cleanup
                 │
                 ▼
 (6) TEACH  src/make_finetune_data.py + src/finetune.py:  LoRA fine-tuning on 498 receipts
            pick the better version on dev100 (100 practice receipts never used for teaching)
                 │
                 ▼
 (7) TEST   run once on the 370 test receipts ──► results/test_final/  =  96.22%
```

## One receipt followed through every chapter

To make things concrete, we follow one real receipt, **receipt2**, a gift-shop receipt. Its image is `train/receipt_images/receipt2.jpg` and its label is `train/receipt_labeled_text/receipt2.json`.

In each chapter you will see what happened to it: its label, what each version of the model answered, and how the answer was scored.

| Version | What it got wrong on receipt2 |
|---|---|
| Model as it is (prompt v1) | 4 fields wrong: a number from the "Room No" line taken as the tax number, a product code left in an item name, the total before rounding, and the rounding taken as tax |
| Better instructions (prompt v2) | 1 field wrong: still the "Room No" number as the tax number |
| Fine-tuned (final) | no field wrong under the headline rule |

Both versions without fine-tuning also misread one letter of an item name ("CARD/BD" for "CARD/GD"). The scorer forgives that, because item names only need to be 95% similar ([chapter 4](04_step2_scoring.md)).

## Where things are in the project

| Folder or file | What it holds |
|---|---|
| `train/`, `test/` | receipt images and their labels |
| `src/` | the code and the instruction texts (prompts) |
| `outputs/<run>/` | what the model answered, one JSON per receipt |
| `results/<run>/` | the scores of that run |
| `metadata/` | the record of every label correction, and mapping files |
| `models/` | the model weights and the fine-tuned adapter (big files, not on GitHub) |
| `finetune/` | the training examples (rebuilt by a script, not on GitHub) |
| `README.md` | the technical summary of the project |
| `requirements.txt` | the exact software versions needed |
