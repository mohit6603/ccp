"""Run Qwen3-VL-4B on a folder of receipt images -> one JSON file per image.

Output layout (names match the input image names):
    <out>/receiptN.json      parsed + schema-checked prediction (only written when valid)
    <out>/raw/receiptN.txt   raw model text for every image
    <out>/timing.csv         seconds per image
    <out>/run_config.json    everything needed to reproduce the run

The run is resumable: images that already have a raw file are skipped.
Test labels are never read by this script.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import transformers
from PIL import Image, ImageOps
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

KEYS = ["company", "date", "address", "tax_number", "line_items", "total", "tax_amount", "service_charge"]
ITEM_KEYS = ["description", "quantity", "unit_price", "amount"]
PATCH = 32  # Qwen3-VL: 16px patch x 2x2 merge


def natural_key(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.stem)]


def load_image(path, max_pixels):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    scale = min(1.0, (max_pixels / (w * h)) ** 0.5)
    w2 = max(PATCH, round(w * scale / PATCH) * PATCH)
    h2 = max(PATCH, round(h * scale / PATCH) * PATCH)
    return img.resize((w2, h2), Image.BICUBIC) if (w2, h2) != (w, h) else img


def parse_output(text):
    """Return (dict, None) for a valid schema-conforming answer, else (None, reason)."""
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        d = json.loads(s)
    except json.JSONDecodeError as e:
        return None, f"json_error: {e}"
    if not isinstance(d, dict) or set(d) != set(KEYS):
        return None, "schema_error: keys"
    items = d["line_items"]
    if items is not None:
        if not isinstance(items, list) or not all(isinstance(i, dict) and set(i) == set(ITEM_KEYS) for i in items):
            return None, "schema_error: line_items"
    return {k: d[k] if k != "line_items" or d[k] is None else [{ik: i[ik] for ik in ITEM_KEYS} for i in d[k]] for k in KEYS}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder with receiptN.jpg")
    ap.add_argument("--out", required=True, help="output run folder, e.g. outputs/train_baseline")
    ap.add_argument("--model", default="models/Qwen3-VL-4B-Instruct")
    ap.add_argument("--prompt-file", default="src/prompt_v1.txt")
    ap.add_argument("--max-pixels", type=int, default=1024 * PATCH * PATCH)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"],
                    help="default: float16 on a CUDA GPU (T4 has no native bf16), bfloat16 on CPU")
    ap.add_argument("--names", nargs="*", help="only these image base names, e.g. receipt2 receipt7")
    ap.add_argument("--limit", type=int, help="only the first N images (natural order)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.dtype = args.dtype or ("float16" if device == "cuda" else "bfloat16")
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    prompt = Path(args.prompt_file).read_text().strip()

    images = sorted(Path(args.images).glob("*.jpg"), key=natural_key)
    if args.names:
        images = [p for p in images if p.stem in set(args.names)]
    if args.limit:
        images = images[: args.limit]

    cfg_path = out / "run_config.json"
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps({
            "model": args.model, "prompt_file": args.prompt_file,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "max_pixels": args.max_pixels, "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype, "decoding": "greedy (do_sample=False)",
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "torch": torch.__version__, "transformers": transformers.__version__,
            "started": datetime.now().isoformat(timespec="seconds"),
        }, indent=2))

    todo = [p for p in images if not (out / "raw" / f"{p.stem}.txt").exists()]
    print(f"{len(images)} images selected, {len(todo)} to run", flush=True)
    if not todo:
        return

    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    timing_path = out / "timing.csv"
    new_file = not timing_path.exists()
    with open(timing_path, "a", newline="") as tf:
        tw = csv.writer(tf)
        if new_file:
            tw.writerow(["name", "seconds", "image_size", "input_tokens", "output_tokens", "valid", "error"])
        for n, p in enumerate(todo, 1):
            t0 = time.time()
            img = load_image(p, args.max_pixels)
            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            new_tokens = gen[0][inputs["input_ids"].shape[1]:]
            text = processor.decode(new_tokens, skip_special_tokens=True)

            (out / "raw" / f"{p.stem}.txt").write_text(text)
            parsed, err = parse_output(text)
            if parsed is not None:
                (out / f"{p.stem}.json").write_text(json.dumps(parsed, indent=4, ensure_ascii=False))
            dt = time.time() - t0
            tw.writerow([p.stem, f"{dt:.1f}", f"{img.size[0]}x{img.size[1]}",
                         inputs["input_ids"].shape[1], len(new_tokens), parsed is not None, err or ""])
            tf.flush()
            print(f"[{n}/{len(todo)}] {p.stem} {dt:.1f}s valid={parsed is not None}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
