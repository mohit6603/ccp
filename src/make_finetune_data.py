"""Build the LoRA fine-tuning data from the train split -> finetune/train.jsonl, finetune/valid.jsonl, finetune/images/.

Usage (from the project folder):
    .venv/bin/python src/make_finetune_data.py

- train.jsonl: train receipts minus the dev100 subset and minus the three train receipts that have a
  duplicate in the test split (receipt443, receipt796, receipt729), so test scores stay honest.
- valid.jsonl: the dev100 subset, used only to watch validation loss. dev100 is never trained on.
- Each row: the receipt image (resized exactly as run_model.py does), the prompt as the user message,
  and the label JSON as the assistant answer. Test labels are never read.
"""
import argparse
import json
from pathlib import Path

from run_model import KEYS, load_image, natural_key

TEST_DUPLICATES = {"receipt443", "receipt796", "receipt729"}  # same receipt as test 442, 249, 746


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-file", default="src/prompt_v1.txt")
    ap.add_argument("--max-pixels", type=int, default=1024 * 32 * 32)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = root / "finetune"
    (out / "images").mkdir(parents=True, exist_ok=True)
    prompt = (root / args.prompt_file).read_text().strip()
    dev = set((root / "src" / "dev100.txt").read_text().split())

    rows = {"train": [], "valid": []}
    for p in sorted((root / "train" / "receipt_images").glob("*.jpg"), key=natural_key):
        if p.stem in TEST_DUPLICATES:
            continue
        label = json.loads((root / "train" / "receipt_labeled_text" / f"{p.stem}.json").read_text())
        img_rel = f"images/{p.stem}.png"
        if not (out / img_rel).exists():
            load_image(p, args.max_pixels).save(out / img_rel)
        answer = json.dumps({k: label[k] for k in KEYS}, ensure_ascii=False)
        rows["valid" if p.stem in dev else "train"].append({
            "images": [img_rel],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
        })
    for split, items in rows.items():
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for r in items:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(items)} receipts")
    (out / "info.json").write_text(json.dumps({
        "prompt_file": args.prompt_file, "max_pixels": args.max_pixels,
        "excluded_test_duplicates": sorted(TEST_DUPLICATES), "valid": "src/dev100.txt",
    }, indent=2))


if __name__ == "__main__":
    main()
