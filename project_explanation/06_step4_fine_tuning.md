# 6. Step 4: fine-tuning, or teaching by examples

## 6.1 The idea

**Fine-tuning** means continuing to train an already-trained model on our own examples, so it gets better at our specific task.

Think of a new employee. You can give them a rulebook (that is the prompt, chapter 5), or you can sit them down with 500 receipts that were already filled in correctly, and let them learn the pattern. Fine-tuning is the second way.

Each example is:
- **the receipt image** and **the short instruction** (`src/prompt_ft.txt`);
- **the correct answer**, which is the label JSON.

## 6.2 LoRA, in simple words

The model has about 4 billion numbers (parameters). Training all of them would need far more memory than a 16 GB laptop has.

**LoRA** (Low-Rank Adaptation) solves this:

1. **Freeze** all 4 billion original numbers. They do not change.
2. **Add small "correction" layers** next to some of the model's internal layers. Each correction is two thin tables of numbers. Information passes through a narrow middle of only 16 numbers, and this width is called the **rank**.
3. **Train only the corrections.**

```
            ┌──────────────────────────────┐
 input ──┬─►│ original layer (frozen)       │──┐
         │  └──────────────────────────────┘  ├─(+)──► output
         └─► thin table A ─► 16 ─► thin table B ┘
              (these two are trained)
```

| | |
|---|---|
| Where the corrections go | 7 parts of each of the 36 language-model layers (q, k, v, o, gate, up, down projections) = 252 places |
| Rank | 16 |
| Strength (alpha) | 32, which means the correction is multiplied by 32 ÷ 16 = 2 |
| Numbers trained | 33,030,144, which is **0.74%** of the model |
| Vision encoder | not changed |
| Saved file | `adapters.safetensors`, 132 MB. This file is called the **adapter**. |

When the model runs, it uses the base model plus the adapter (`--adapter models/lora_v1/epoch2`).

## 6.3 Building the training examples (`src/make_finetune_data.py`)

```bash
.venv/bin/python src/make_finetune_data.py --prompt-file src/prompt_ft.txt
```

What the script does:
1. Go through the 600 train receipts.
2. Skip receipt443, receipt796 and receipt729, the three with near-duplicates in test (chapter 1.6).
3. Save each image already shrunk exactly as `run_model.py` shrinks it, as `finetune/images/receiptN.png`.
4. Write the answer as the label in compact JSON, with the keys in the standard order.
5. Put dev100 receipts in `finetune/valid.jsonl` and all others in `finetune/train.jsonl`.

| File | Receipts | Used for |
|---|---|---|
| `finetune/train.jsonl` | 498 (600 − 100 dev100 − 2 duplicates not in dev100) | teaching |
| `finetune/valid.jsonl` | 99 (dev100 minus receipt729, a duplicate) | checking progress only, never teaching |

One row of `train.jsonl` looks like this (shortened):

```json
{"images": ["images/receipt5.png"],
 "messages": [{"role": "user", "content": "Extract the information from this receipt as one JSON object with the keys company, date, ..."},
              {"role": "assistant", "content": "{\"company\": \"...\", \"date\": \"...\", ...}"}]}
```

**Why such a short instruction?** The model learns the rules from the examples themselves, so it does not need a long rulebook. The short prompt also makes every receipt faster to process. Receipt2's input is 446 tokens with the short prompt, against 1,359 with prompt v2.

## 6.4 How training works (`src/finetune.py`)

```bash
.venv/bin/python src/finetune.py --out models/lora_v1 --epochs 2
```

**1. Prepare.** Load the 8-bit model, add the LoRA layers, and turn on memory saving (section 6.5).

**2. Show one example.** The model reads the image, the instruction and the **correct** answer. At every position of the answer, it predicts the next token.

**3. Measure the error.** The **loss** is a number that says how surprised the model was by the correct tokens. 0 means not surprised at all.
- It is measured **only on the answer tokens**, not on the image or the instruction.
- A handy way to read it: e^(−loss) is the average probability the model gave to the correct next token. For example, a loss of 0.21 means about 81%, and a loss of 0.018 means about 98%.

**4. Find the direction to improve.** For each of the 33 million LoRA numbers, the computer works out whether making it slightly bigger or smaller would lower the loss. This is called the **gradient**.

**5. Collect 4 examples, then update.** The gradients of 4 receipts are averaged, and then the LoRA numbers are moved a small step in the better direction. That is one **optimizer step**, done by an optimizer called **AdamW**. Collecting several examples before one update is called **gradient accumulation**. It gives the effect of a batch of 4 while only holding 1 receipt in memory.

**6. Step size.** The **learning rate** controls how big each step is:
- it starts small and grows over the first 10 steps (the **warm-up**), up to 0.0001;
- it then slowly shrinks along a smooth curve (a **cosine schedule**) down to 0.00001 by the end.

**7. Repeat.** One pass over all 498 receipts is an **epoch**: 125 steps, because the last step has only 2 receipts. We ran 2 epochs, 250 steps in total, with the receipts shuffled in a different order each epoch.

**8. Check after each epoch.** Compute the loss on the 99 dev receipts, without any learning. This is the **validation loss**. Then save the adapter (`models/lora_v1/epoch1/`, `epoch2/`).

**9. Safety copies.** Every 25 steps a copy is saved in `models/lora_v1/last/`, in case the run stops.

Everything is logged in `models/lora_v1/train_log.csv`: the loss of each step, the learning rate, the time and the memory.

## 6.5 Fitting it into 16 GB

Training needs much more memory than running, because the computer must remember intermediate results to compute gradients. Three tricks were used:

- **Gradient checkpointing:** instead of storing every intermediate result, recompute some of them when needed. It is slower, but uses far less memory.
- **Scoring only the answer positions:** the model's final step turns each position into scores for all 151,936 possible tokens. We compute that only for the answer tokens, not for the roughly 1,000 image and instruction tokens. It gives the same loss with much less memory.
- **One receipt at a time,** with gradient accumulation (step 5 above).

Even so, memory peaked at **17.2 GB**, more than the Mac's 16 GB. macOS used **swap**: it moved some memory to the disk, which is slower but works. Each step (4 receipts) took about 97 seconds, and the whole training took **about 7 hours**. The Mac was kept awake with `caffeinate`.

## 6.6 What happened during training

| Moment | Loss | e^(−loss) |
|---|---|---|
| Step 1 (start) | 0.209 | 81% |
| Step 10 | 0.029 | 97% |
| End of epoch 1 (training receipts) | mostly 0.01 to 0.05 (it jumps around from step to step) | about 97% |
| **Validation after epoch 1** (dev receipts) | **0.0224** | 97.8% |
| End of epoch 2 (training receipts) | about 0.002 to 0.02 | about 99% |
| **Validation after epoch 2** (dev receipts) | **0.0181** | 98.2% |

**Overfitting** means the model memorises its training examples instead of learning the general pattern. The sign is a validation loss that goes **up** while the training loss keeps going down. Here the validation loss went **down** from epoch 1 to epoch 2, so the model was still learning things that also hold for new receipts.

## 6.7 Choosing the version with dev100

Both adapters were run on the 100 dev receipts and scored:

| Version | Overall | Exact | 95% similar | Line-item F1 |
|---|---|---|---|---|
| Prompt v2, no fine-tuning (for comparison) | 91.86% | 87.86% | 94.43% | 47.99% |
| Fine-tuned, epoch 1 | 96.14% | 92.43% | 97.14% | 80.14% |
| **Fine-tuned, epoch 2** | **97.29%** | **94.14%** | **98.14%** | **84.98%** |

Epoch 2 was better on every measure, so it became the final model. The test receipts were not used for this choice.

How each field changed on dev100 (overall rule):

| Field | Prompt v2 | Fine-tuned epoch 2 |
|---|---|---|
| company | 86% | 98% |
| date | 97% | 98% |
| address | 83% | 94% |
| tax_number | 87% | 95% |
| total | 93% | 99% |
| tax_amount | 97% | 98% |
| service_charge | 100% | 99% |

## 6.8 Receipt2 after fine-tuning

The model writes its answer on one line. It is spread over several lines here so it is easier to read:

```json
{"company": "INDAH GIFT & HOME DECO", "date": "19/10/2018",
 "address": "27,JALAN DEDAP 13, TAMAN JOHOR JAYA, 81100 JOHOR BAHRU,JOHOR.", "tax_number": null,
 "line_items": [{"description": "ST-PRIVILEGE CARD/GD INDAH", "quantity": 1, "unit_price": "10.00", "amount": "10.00"},
                {"description": "GF-TABLE LAMP/STITCH <I>", "quantity": 1, "unit_price": "55.90", "amount": "55.90"}],
 "total": "60.30", "tax_amount": null, "service_charge": null}
```

What changed:
- **Every field is right.** The tax number is now correctly null, and "CARD/**G**D" is read correctly.
- **The address is copied exactly as printed** ("27,JALAN"). It differs from the label only in the spaces after commas, so it counts as correct under the overall rule.
- **The answer is now one compact line**, like the training answers, so it is shorter: 190 tokens instead of 241. It took 15.9 seconds.

## 6.9 Checks that made sure nothing went wrong

- **Same format in training and running.** We checked in the code that the fine-tuned model receives exactly the same prompt text, the same chat layout (image first) and the same image size at run time as during training.
- **The adapter really loads.** The adapter check confirmed that all 504 adapter weights load (chapter 3.3).
- **No help from the test.** The near-duplicates of test receipts were kept out of training, and dev100 was never taught.
