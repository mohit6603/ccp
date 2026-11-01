# 8. Replicating the project and using it

## 8.1 What runs where

| Task | Apple Silicon Mac (M1 or newer) | Ubuntu laptop, Intel CPU, no graphics card (your office laptop) | Windows |
|---|---|---|---|
| Install from `requirements.txt` | yes (tested) | yes: every package has a ready Linux build for Python 3.13 | scoring part only |
| Recompute every score (`evaluate.py`) | yes, seconds | yes, seconds | yes |
| Study outputs, results and logs | yes | yes | yes |
| Run the model (`run_model.py`) | yes, about 20 s per receipt | starts, but far too slow to use: CPU only, one core (section 8.4) | no |
| Fine-tune (`finetune.py`) | yes, about 7 hours (16 GB memory) | not practical: it would take weeks on a CPU | no |

Why the difference: the model runs on **MLX**. MLX uses the fast graphics processor (GPU) only on Apple Silicon Macs. On Linux it has a CPU-only version. That version gives the same answers in principle, but it is far slower.

## 8.2 Setup on an Apple Silicon Mac

```bash
git clone https://github.com/mohit6603/ccp.git
cd ccp
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then get the model files, choosing one of two ways:
- **A. Copy them from the original Mac** (simplest, and guaranteed identical):
  - `models/Qwen3-VL-4B-Instruct-8bit/` (4.8 GB);
  - `models/lora_v1/epoch1/` and `models/lora_v1/epoch2/` (132 MB each).
- **B. Download and convert:** the commands are in the README's Setup section. You still need to copy the adapter folders, or retrain them (about 7 hours).

Check the model file:

```bash
shasum -a 256 models/Qwen3-VL-4B-Instruct-8bit/model.safetensors
# must print b3d9193e10e795bc09572ddbf779c3cdb9e1d7e6c42556f5e15d0d048bf4e55f
```

`requirements.txt` was verified on 2026-11-01: a brand-new environment built from it had exactly the same 75 packages, reproduced the scores exactly, and gave a byte-identical model answer for receipt2.

## 8.3 Setup on your Ubuntu office laptop

You need Ubuntu 22.04 or newer, because MLX's Linux version needs it.

```bash
# 1. Python 3.13 and git
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.13 python3.13-venv git

# 2. The project
git clone https://github.com/mohit6603/ccp.git
cd ccp
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt      # picks the CPU version of MLX automatically
```

**3. The model files (only needed to run the model).** Copy these from the Mac with a USB drive or a private Google Drive folder, into the same paths:
- `models/Qwen3-VL-4B-Instruct-8bit/`
- `models/lora_v1/epoch1/` and `models/lora_v1/epoch2/`

Don't put them on the public GitHub. The repository is public, and the dataset's terms of use have not been confirmed. Then check the model file:

```bash
sha256sum models/Qwen3-VL-4B-Instruct-8bit/model.safetensors
# must print b3d9193e10e795bc09572ddbf779c3cdb9e1d7e6c42556f5e15d0d048bf4e55f
```

## 8.4 Running the model on the Ubuntu laptop: what to expect

The same command works on Ubuntu, and MLX uses the CPU automatically:

```bash
.venv/bin/python src/run_model.py --split test --run cpu_try --names receipt1 --adapter models/lora_v1/epoch2 --prompt-file src/prompt_ft.txt
```

Then compare `outputs/cpu_try/receipt1.json` with `outputs/test_final/receipt1.json`.

**How slow is it?** We imitated your laptop on the Mac: MLX forced onto the processor, with the Mac's graphics processor hidden.
- **It started correctly:** it loaded the model and passed the adapter check.
- **It was far too slow:** MLX used **only one processor core**. After **36 minutes of processor time it still had not finished the first receipt**, so we stopped it. The Mac's GPU needs 16 seconds for the same receipt.
- **Your i5 is likely to be about as slow or slower.**

So on that laptop, running the model with MLX is **not practical**, not even for a quick demonstration. Use the laptop for the scoring, the checks and studying the outputs, and see the options below for running the model.

**If you need to run the model at the office at a useful speed**, the options are:
1. **Run the model on the Mac** (or any Apple Silicon Mac), and use the office laptop for scoring, studying and reporting. Nothing new is needed for this.
2. **A PyTorch version for the CPU.** PyTorch is another AI library that can use all processor cores. An earlier version of this project measured 4 to 6 minutes per receipt with PyTorch on this same i5 laptop, before fine-tuning: 251 seconds in a first test, and about 370 seconds on average over 10 receipts. This version doesn't exist yet. It would need the adapter converted to PyTorch's format, and its answers would differ slightly, so it would need its own dev100 check.
3. **A machine with an NVIDIA graphics card, or Google Colab.** It is fast, but it needs a PyTorch version too, and uploading receipts to Colab needs company approval.

## 8.5 Recomputing the scores (any computer)

This works everywhere, needs no model, and takes seconds:

```bash
.venv/bin/python src/evaluate.py --split test --run test_final --res-dir results_check/test_final
diff results_check/test_final/summary.csv results/test_final/summary.csv && echo "identical"
```

`--res-dir` writes the new scores to a separate folder, so the original `results/test_final/` stays untouched. On Windows, use `.venv\Scripts\python` and `fc` instead of `diff`.

## 8.6 Re-running the final model (Apple Silicon Mac)

```bash
.venv/bin/python src/run_model.py --split test --run test_final_check --adapter models/lora_v1/epoch2 --prompt-file src/prompt_ft.txt
.venv/bin/python src/evaluate.py --split test --run test_final_check
```

Use a **new run name**. `outputs/test_final/` is already complete, so re-running that name runs nothing. With the same files and versions the answers should be the same, or almost the same: a different chip or macOS version can change a few answers slightly.

## 8.7 Using it on new receipts

```bash
.venv/bin/python src/run_model.py --images-dir ~/Desktop/new_receipts --run new_receipts --adapter models/lora_v1/epoch2 --prompt-file src/prompt_ft.txt
```

- **Input:** any folder of JPG, PNG, WEBP, BMP or TIFF images.
- **Output:** the answers in `outputs/new_receipts/`.
- **No scoring:** there are no labels, so there is nothing to score.

## 8.8 Fixing common problems

| Message or problem | What it means | What to do |
|---|---|---|
| `run_config differs from the existing run in [...]; use a new --run name` | You changed a setting, but the run folder already holds answers made with other settings | Use a new `--run` name, or the same settings as before |
| `0 to run` | Every receipt in that run already has an answer | Nothing; or use a new run name to run again |
| `adapter not loaded: ...` | The adapter files are missing, broken, or from another model | Copy `models/lora_v1/epoch2/` again and check that both files are there |
| An error about the model path, or about a Hugging Face "repo id", when the model loads | The model files are not in `models/Qwen3-VL-4B-Instruct-8bit/` | Copy or convert them (sections 8.2 and 8.3) |
| An image is not picked up | Its format is not read (for example iPhone HEIC) | Convert it to JPG or PNG |
| The Mac went to sleep and the run stopped | Long runs need the Mac awake | Start it as `caffeinate -s -i <command>` with the charger in and the lid open, then re-run the same command; it continues |
| `valid=False json_error` for a receipt | The model wrote broken JSON (like receipt81) | Look at `outputs/<run>/raw/receiptN.txt`; the receipt counts as wrong when scoring |
| Very slow, or the computer freezes | Not enough memory: running needs about 7 GB free, training about 17 GB | Close other programs; on Ubuntu, see section 8.4 |
