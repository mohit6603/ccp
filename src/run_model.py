"""Run Qwen3-VL-4B (MLX, on this Mac) on receipt images -> one JSON file per image.

Usage (from the project folder):
    .venv/bin/python src/run_model.py --split train --run train_baseline
    .venv/bin/python src/run_model.py --split train --run trial_5 --limit 5
    .venv/bin/python src/run_model.py --split test --run test_final --adapter models/lora_v1/epoch2 --prompt-file src/prompt_ft.txt
    .venv/bin/python src/run_model.py --images-dir ~/Desktop/new_receipts --run new_receipts --adapter models/lora_v1/epoch2 --prompt-file src/prompt_ft.txt

Output layout (file names match the input image names):
    outputs/<run>/receiptN.json      parsed, schema-checked, cleaned prediction (written only when valid;
                                     --no-clean skips clean_values, as in train_baseline)
    outputs/<run>/raw/receiptN.txt   raw model text for every image
    outputs/<run>/timing.csv         seconds, tokens and validity per image
    outputs/<run>/run_config.json    everything needed to reproduce the run

The run is resumable: images that already have a raw file are skipped.
Labels are never read by this script.
"""
import argparse
import csv
import hashlib
import json
import platform
import re
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from PIL import Image, ImageOps

KEYS = ["company", "date", "address", "tax_number", "line_items", "total", "tax_amount", "service_charge"]
ITEM_KEYS = ["description", "quantity", "unit_price", "amount"]
PATCH = 32  # Qwen3-VL: 16 px patch x 2x2 merge = one image token per 32x32 px
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}  # for --images-dir

Image.MAX_IMAGE_PIXELS = None  # some receipts are 4961x7016


def natural_key(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.stem)]


def load_image(path, max_pixels):
    """Upright RGB image, scaled down (never up) to at most max_pixels, sides multiples of 32."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    scale = min(1.0, (max_pixels / (w * h)) ** 0.5)
    w2 = max(PATCH, int(w * scale / PATCH) * PATCH)
    h2 = max(PATCH, int(h * scale / PATCH) * PATCH)
    return img.resize((w2, h2), Image.BICUBIC) if (w2, h2) != (w, h) else img


def check_adapter(model, path):
    """Stop unless every tensor of the adapter file is in the model with its saved values
    (mlx-vlm loads adapters with strict=False, which would silently skip a name that does not match)."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    saved = mx.load(str(path))
    params = dict(tree_flatten(model.parameters()))
    bad = [k for k, v in saved.items() if k not in params or params[k].shape != v.shape
           or not mx.allclose(params[k], v, rtol=1e-2, atol=1e-6).item()]
    if bad:
        raise SystemExit(f"adapter not loaded: {len(bad)} of {len(saved)} tensors missing or different, e.g. {bad[0]}")
    print(f"adapter check: all {len(saved)} tensors of {path} are in the model", flush=True)


def parse_output(text, clean=False):
    """Return (dict, None) for a valid answer, else (None, reason).
    clean=False: strict, no repair (train_baseline). clean=True: missing keys become null and
    clean_values() fixes formatting only (see its docstring)."""
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        d = json.loads(s)
    except json.JSONDecodeError as e:
        return None, f"json_error: {e}"
    if clean and isinstance(d, dict):
        d = {k: d.get(k) for k in KEYS}
        if isinstance(d["line_items"], list):
            d["line_items"] = [{k: i.get(k) for k in ITEM_KEYS} if isinstance(i, dict) else i for i in d["line_items"]]
    if not isinstance(d, dict) or set(d) != set(KEYS):
        return None, "schema_error: keys"
    items = d["line_items"]
    if items is not None and not (isinstance(items, list) and all(isinstance(i, dict) and set(i) == set(ITEM_KEYS) for i in items)):
        return None, "schema_error: line_items"
    out = {k: d[k] for k in KEYS}
    if items is not None:
        out["line_items"] = [{k: i[k] for k in ITEM_KEYS} for i in items]
    return (clean_values(out) if clean else out), None


def _null(v):
    return v is None or (isinstance(v, str) and v.strip().lower() in ("", "null", "none"))


def _amount(v):
    """'RM6.69' -> '6.69', '4.00SR' -> '4.00', '$8.20' -> '8.20', '1,234.50' -> '1234.50'; anything else unchanged."""
    if _null(v):
        return None
    s = str(v).strip()
    m = re.fullmatch(r"(-?)\s*(?:RM|MYR|\$)?\s*(-?\d+(?:\.\d+)?)\s*(?:SR|ZR|[A-Z])?", s.replace(",", "").upper())
    return (m.group(1) + m.group(2)).replace("--", "-") if m else s


def clean_values(d):
    """Formatting-only cleanup, the same for every receipt; never reads labels. No train label contains
    any of the removed text (currency, tax flags, 'GST REG #' prefixes, '(123456-X)' registration numbers,
    '@ price' or '##' in item names), so this only removes formatting the labels never use."""
    out = {}
    for k in KEYS:
        v = d[k]
        if k == "line_items":
            out[k] = v
        elif _null(v):
            out[k] = None
        elif k in ("total", "tax_amount", "service_charge"):
            out[k] = _amount(v)
        else:
            s = re.sub(r"\s+", " ", str(v)).strip()
            if k == "company":
                s = re.sub(r"\s*\(\s*\d{3,}[-\s]?[A-Z]?\s*\)", "", s).strip()
            if k == "tax_number" and re.search(r"\d", s):
                s = re.sub(r"^\D+", "", s)
                s = re.sub(r"\s*\)$", "", s)
            out[k] = s
    if isinstance(out["line_items"], list):
        items = []
        for it in out["line_items"]:
            desc = None if _null(it["description"]) else str(it["description"])
            if desc:
                stripped = re.sub(r"\s*@.*$", "", desc).replace("##", "").strip()
                desc = stripped or desc
            q = it["quantity"]
            if isinstance(q, str):
                q = None if _null(q) else (int(float(q)) if re.fullmatch(r"-?\d+(\.0+)?", q.strip()) else
                                           float(q) if re.fullmatch(r"-?\d*\.\d+", q.strip()) else q)
            items.append({"description": desc, "quantity": q,
                          "unit_price": _amount(it["unit_price"]), "amount": _amount(it["amount"])})
        out["line_items"] = items
    return out


def main():
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--split", choices=["train", "test"])
    source.add_argument("--images-dir", help="any folder of receipt images, no labels needed, e.g. ~/Desktop/new_receipts")
    ap.add_argument("--run", required=True, help="run name, e.g. train_baseline -> outputs/train_baseline/")
    ap.add_argument("--model", default="models/Qwen3-VL-4B-Instruct-8bit")
    ap.add_argument("--adapter", help="fine-tuned LoRA adapter folder, e.g. models/lora_v1")
    ap.add_argument("--prompt-file", default="src/prompt_v1.txt")
    ap.add_argument("--max-pixels", type=int, default=1024 * PATCH * PATCH, help="default: about 1024 image tokens")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--names", nargs="*", help="only these receipts, e.g. receipt2 receipt7")
    ap.add_argument("--names-file", help="only the receipts listed in this file, e.g. src/dev100.txt")
    ap.add_argument("--limit", type=int, help="only the first N receipts (natural order)")
    ap.add_argument("--no-clean", action="store_true", help="strict parsing without clean_values() (as in train_baseline)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = root / "outputs" / args.run
    (out / "raw").mkdir(parents=True, exist_ok=True)
    prompt = (root / args.prompt_file).read_text().strip()

    if args.images_dir:
        folder = Path(args.images_dir).expanduser().resolve()
        images = sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_TYPES), key=natural_key)
    else:
        folder = None
        images = sorted((root / args.split / "receipt_images").glob("*.jpg"), key=natural_key)
    if args.names_file:
        args.names = (args.names or []) + (root / args.names_file).read_text().split()
    if args.names:
        images = [p for p in images if p.stem in set(args.names)]
    if args.limit:
        images = images[: args.limit]

    cfg_path = out / "run_config.json"
    cfg = {
        "split": args.split, "images_dir": str(folder) if folder else None,
        "model": args.model, "adapter": args.adapter, "prompt_file": args.prompt_file,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "max_pixels": args.max_pixels, "max_tokens": args.max_tokens,
        "decoding": "greedy (temperature 0)", "runtime": "mlx-vlm on Apple Silicon",
        "machine": platform.platform(), "mlx": version("mlx"), "mlx_vlm": version("mlx-vlm"),
        "postprocess": "none (strict)" if args.no_clean else "clean_values",
    }
    if cfg_path.exists():  # resuming: settings must not change mid-run
        old = json.loads(cfg_path.read_text())
        changed = [k for k in cfg if old.get(k) != cfg[k]]
        if changed:
            raise SystemExit(f"run_config differs from the existing run in {changed}; use a new --run name")
    else:
        cfg["started"] = datetime.now().isoformat(timespec="seconds")
        cfg_path.write_text(json.dumps(cfg, indent=2))

    todo = [p for p in images if not (out / "raw" / f"{p.stem}.txt").exists()]
    print(f"{len(images)} images selected, {len(todo)} to run", flush=True)
    if not todo:
        return

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(str(root / args.model), adapter_path=str(root / args.adapter) if args.adapter else None)
    if args.adapter:
        check_adapter(model, root / args.adapter / "adapters.safetensors")
    chat = apply_chat_template(processor, model.config, prompt, num_images=1)

    timing_path = out / "timing.csv"
    new_file = not timing_path.exists()
    with open(timing_path, "a", newline="") as tf:
        tw = csv.writer(tf)
        if new_file:
            tw.writerow(["name", "seconds", "image_size", "prompt_tokens", "output_tokens", "peak_memory_gb", "valid", "error"])
        for n, p in enumerate(todo, 1):
            t0 = time.time()
            img = load_image(p, args.max_pixels)
            res = generate(model, processor, chat, image=[img], max_tokens=args.max_tokens, temperature=0.0, verbose=False)
            text = res.text
            (out / "raw" / f"{p.stem}.txt").write_text(text)
            parsed, err = parse_output(text, clean=not args.no_clean)
            if parsed is not None:
                (out / f"{p.stem}.json").write_text(json.dumps(parsed, indent=4, ensure_ascii=False) + "\n")
            if err is None and res.generation_tokens >= args.max_tokens:
                err = "hit max_tokens"
            dt = time.time() - t0
            tw.writerow([p.stem, f"{dt:.1f}", f"{img.size[0]}x{img.size[1]}", res.prompt_tokens,
                         res.generation_tokens, f"{res.peak_memory:.2f}", parsed is not None, err or ""])
            tf.flush()
            print(f"[{n}/{len(todo)}] {p.stem} {dt:.1f}s valid={parsed is not None} {err or ''}", flush=True)


if __name__ == "__main__":
    main()
