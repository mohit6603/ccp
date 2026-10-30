"""Rebuild the Colab upload package: colab/VLMOCR_train.zip, colab/VLMOCR_test.zip, colab/run_on_colab.ipynb.

Zips contain code + images only (no labels). Run from anywhere: python src/make_colab_package.py
"""
import json, zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "colab"

def make_zip(name, files):
    with zipfile.ZipFile(out / name, "w", zipfile.ZIP_STORED) as z:   # jpg is already compressed
        for f in files:
            z.write(root / f, f)

src = ["src/inference.py", "src/prompt_v1.txt"]
train = sorted(str(p.relative_to(root)) for p in (root / "train/receipt_images").glob("*.jpg"))
test = sorted(str(p.relative_to(root)) for p in (root / "test/receipt_images").glob("*.jpg"))
make_zip("VLMOCR_train.zip", src + train)
make_zip("VLMOCR_test.zip", test)

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip().splitlines(True)}

cells = [
 md("""
# Qwen3-VL-4B receipt extraction - run on Colab T4
**Before starting:** Runtime > Change runtime type > **T4 GPU**.
Put `VLMOCR_train.zip` (and later `VLMOCR_test.zip`) in your Google Drive folder `MyDrive/VLMOCR/`.
Outputs are written straight to Drive (`MyDrive/VLMOCR/outputs/<run>/`), so nothing is lost if Colab disconnects.
The script is resumable: after a disconnect run cells 2, 3 and the same run cell again - finished receipts are skipped.
"""),
 code("# 1. GPU check - must show a Tesla T4\n!nvidia-smi"),
 code("""# 2. Mount Drive and unpack the code + training images (no labels are uploaded)
from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/VLMOCR'
!mkdir -p /content/work
!unzip -q -o {DRIVE}/VLMOCR_train.zip -d /content/work
%cd /content/work
!ls"""),
 code("# 3. Libraries (Qwen3-VL needs a recent transformers)\n!pip install -q -U \"transformers>=4.57\" accelerate pillow"),
 code("""# 4. SMOKE TEST: 5 training images. Check speed, valid=True and that the JSON looks right.
!python src/inference.py --images train/receipt_images --out {DRIVE}/outputs/smoke_test_gpu --model Qwen/Qwen3-VL-4B-Instruct --limit 5
import pandas as pd
print(pd.read_csv(f'{DRIVE}/outputs/smoke_test_gpu/timing.csv'))
# If the output is garbage / empty (fp16 overflow), re-run with:  --dtype bfloat16"""),
 code("""# 5. BASELINE on all 600 training images (resumable)
!python src/inference.py --images train/receipt_images --out {DRIVE}/outputs/train_baseline --model Qwen/Qwen3-VL-4B-Instruct"""),
 md("""
---
## FINAL TEST - do NOT run until the configuration is frozen
Only after the baseline/improvement work is finished and we agree the final settings.
Upload `VLMOCR_test.zip` to Drive first (it contains images only, no labels).
"""),
 code("""# 6. FINAL TEST RUN (370 test images) - run only when told to
# !unzip -q -o {DRIVE}/VLMOCR_test.zip -d /content/work
# !python src/inference.py --images test/receipt_images --out {DRIVE}/outputs/test_final --model Qwen/Qwen3-VL-4B-Instruct"""),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 5}
(out / "run_on_colab.ipynb").write_text(json.dumps(nb, indent=1))
print(len(train), "train images,", len(test), "test images")
