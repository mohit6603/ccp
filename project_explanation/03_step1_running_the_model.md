# 3. Step 1: running the model (`src/run_model.py`)

## 3.1 What it does

For every receipt image, the script asks the model for the JSON answer. It saves the answer in a file with the same name as the image.

It **never reads the labels**. It only sees images.

## 3.2 The command

The command that produced the final result:

```bash
.venv/bin/python src/run_model.py --split test --run test_final --adapter models/lora_v1/epoch2 --prompt-file src/prompt_ft.txt
```

`.venv/bin/python` is the Python program inside the project's own environment (`.venv`), where all the needed packages are installed.

| Option | Meaning | Default |
|---|---|---|
| `--split train` or `--split test` | read the images from `train/receipt_images/` or `test/receipt_images/` | (one of these two, or `--images-dir`) |
| `--images-dir FOLDER` | read the images from any folder, for new receipts without labels | |
| `--run NAME` | the name of this run; everything goes to `outputs/NAME/` | (required) |
| `--adapter FOLDER` | use a fine-tuned adapter (chapter 6) | none: the model as it is |
| `--prompt-file FILE` | which instruction text to use | `src/prompt_v1.txt` |
| `--model FOLDER` | which model weights to use | `models/Qwen3-VL-4B-Instruct-8bit` |
| `--max-pixels N` | the image size limit | 1,048,576 pixels (1,024 tiles) |
| `--max-tokens N` | the longest answer allowed | 2,048 tokens |
| `--names-file FILE` | only the receipts listed in this file, for example `src/dev100.txt` | all |
| `--names receipt2 receipt7` | only these receipts | all |
| `--limit N` | only the first N receipts | all |
| `--no-clean` | skip the cleanup step (only the first run, train_baseline, was done like this) | cleanup on |

## 3.3 What happens, step by step

```
 pick images ─► write or check run_config.json ─► skip finished ones ─► load model (+ adapter) once
      │
      └─► for each image:  load + shrink ─► ask model ─► save raw text ─► read JSON + check keys
                                                                              │
                                     save receiptN.json ◄── cleanup ◄─────────┘
                                     add a row to timing.csv
```

**1. Pick the images.** All `.jpg` files of the split, or all images in `--images-dir`, are sorted in "natural" order, so receipt2 comes before receipt10. The `--names`, `--names-file` and `--limit` options then narrow the list.

**2. Write or check `run_config.json`.** This file records every setting of the run:
- the model, the adapter and the prompt file;
- a checksum of the prompt text;
- the image limit and the decoding method;
- the software versions and the date.

If the run folder already exists, the script compares the settings. If anything is different, it stops with *"run_config differs ... use a new --run name"*. That way, two different setups can never be mixed in one folder.

**3. Skip receipts that are already done.** A receipt is done when `raw/receiptN.txt` exists. So if a run is interrupted (power cut, Mac asleep), running the same command again **continues where it stopped**.

**4. Load the model once.** The 8-bit model is loaded, plus the adapter if `--adapter` is given. Then the **adapter check** runs. The library that loads adapters silently skips any weight whose name doesn't match, which would leave you running the plain model without knowing it. So the script checks that all 504 adapter weight files are really inside the model, and stops if not. You will see: *"adapter check: all 504 tensors ... are in the model"*.

**5. Build the chat text once.** The prompt is wrapped in the chat format with one image slot (chapter 2.6).

**6. For each image:**

- **a. Load the image.** Rotate it upright if the photo says it was taken sideways (the EXIF rotation tag), convert it to normal colour (RGB), and shrink it to at most about 1 megapixel, with both sides a multiple of 32.
- **b. Ask the model.** It writes the answer with greedy decoding, up to 2,048 tokens.
- **c. Save the raw text** in `raw/receiptN.txt`, always, even if it is broken.
- **d. Read the JSON.** Remove any markdown fences (```` ```json ````), then read the text as JSON. Then check the shape:
  - there must be exactly the 8 keys;
  - line_items must be null or a list;
  - every item must have exactly description, quantity, unit_price and amount.

  If the shape is wrong, the answer is **invalid**.
- **e. Clean up the format** (section 3.4).
- **f. Save `receiptN.json`**, but only if the answer was valid.
- **g. Add a row to `timing.csv`.**

## 3.4 The cleanup step

The model sometimes writes correct values in a slightly different format from the labels. The cleanup (the `clean_values` function) fixes **format only**, the same way for every receipt:

| What | Example before | After |
|---|---|---|
| currency symbols and tax letters on amounts | `RM6.69`, `4.00SR`, `$8.20` | `6.69`, `4.00`, `8.20` |
| thousands commas in amounts | `1,234.50` | `1234.50` |
| the word "null" or an empty string | `"null"`, `""` | `null` |
| a registration number after the company name | `ABC SDN BHD (123456-X)` | `ABC SDN BHD` |
| label text before the GST number | `GST REG #000800589824` | `000800589824` |
| a price or "##" after an item name | `MILO @1.50`, `##KOPI` | `MILO`, `KOPI` |
| a quantity written as text | `"2"` | `2` |
| line breaks or double spaces inside company, date, address or tax_number | `27,JALAN DEDAP 13,⏎TAMAN JOHOR JAYA` | `27,JALAN DEDAP 13, TAMAN JOHOR JAYA` |
| a missing key | (no `service_charge` key) | `"service_charge": null` |

**Why this is safe:**
- **No train label contains any of the removed text.** That was checked before the cleanup was used.
- **It never looks at a label**, so it cannot "copy" the right answer.
- **The original answer is always kept** in `raw/`.

## 3.5 The output files

```
outputs/test_final/
├── receipt1.json        the cleaned answer (only for valid answers)
├── receipt3.json
├── ...
├── raw/receipt1.txt     the model's exact text, for every receipt
├── timing.csv           one row per receipt
└── run_config.json      all settings of the run
```

The columns of `timing.csv`:

| Column | Meaning | receipt2 (fine-tuned, dev100 run) |
|---|---|---|
| name | the receipt | receipt2 |
| seconds | time for this receipt | 15.9 |
| image_size | size the model saw | 416x992 |
| prompt_tokens | tokens read (image + instruction) | 446 |
| output_tokens | tokens written | 190 |
| peak_memory_gb | the most memory used so far | 6.10 |
| valid | whether the answer was valid JSON with the right keys | True |
| error | why it was invalid, if it was | (empty) |

## 3.6 Running it on new receipts (no labels needed)

Put the images in any folder. JPG, PNG, WEBP, BMP and TIFF are read; convert iPhone HEIC photos to JPG first. Then:

```bash
.venv/bin/python src/run_model.py --images-dir ~/Desktop/new_receipts --run new_receipts --adapter models/lora_v1/epoch2 --prompt-file src/prompt_ft.txt
```

The answer for `shop.png` appears in `outputs/new_receipts/shop.json`.

This was tested with a renamed JPG and a PNG. The answers were identical to the earlier runs on the same receipts, so the file name, folder and format don't change the result.

## 3.7 Every run of this script in this project

| Run name | Receipts | Prompt | Adapter | Cleanup |
|---|---|---|---|---|
| train_baseline | all 600 train | v1 | none | off |
| dev100_prompt_v2 | dev100 | v2 | none | on |
| dev100_lora_v1_ep1 and _ep2 | dev100 | ft | epoch 1 or epoch 2 | on |
| test_final | all 370 test | ft | epoch 2 | on |
