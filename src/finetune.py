"""LoRA fine-tuning of Qwen3-VL-4B (8-bit MLX base) on finetune/train.jsonl, on this Mac.

Usage (from the project folder, after src/make_finetune_data.py):
    .venv/bin/python src/finetune.py --out models/lora_v1 --epochs 2

Writes  <out>/adapters.safetensors + adapter_config.json   final LoRA adapter (used by run_model.py --adapter)
        <out>/epochN/                                      adapter after each epoch
        <out>/last/                                        safety copy every 25 optimizer steps
        <out>/train_log.csv                                loss per optimizer step, validation loss per epoch
        <out>/train_config.json                            settings

The loss is computed on the answer tokens only. To fit in 16 GB, the vocabulary projection is applied only
at those positions instead of at every image and prompt position (same loss, far less memory).
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

from mlx_vlm import load
from mlx_vlm.trainer.common.model import prepare_model_for_training
from mlx_vlm.trainer.common.utils import enable_model_gradient_checkpointing
from mlx_vlm.trainer.core import save_adapter
from mlx_vlm.trainer.datasets.loading import load as load_rows
from mlx_vlm.trainer.vlm.sft.dataset import VisionDataset
from mlx_vlm.trainer.vlm.sft.runtime import _forward_vlm, collate_vlm_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-VL-4B-Instruct-8bit")
    ap.add_argument("--data", default="finetune")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=10, help="optimizer steps of linear warmup")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--max-steps", type=int, help="stop after this many samples (trial runs)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(json.dumps(vars(args), indent=2))
    random.seed(args.seed)
    mx.random.seed(args.seed)

    model, processor = load(str(root / args.model))
    prepare_model_for_training(model, train_type="lora", lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)
    enable_model_gradient_checkpointing(model)

    # The language model returns hidden states; the vocabulary projection is applied in loss_fn
    # only at answer positions.
    embed = model.language_model.model.embed_tokens
    project = embed.as_linear
    embed.as_linear = lambda h: h

    rows = load_rows(str(root / args.data))
    opts = {"train_on_completions": True}
    train_ds = VisionDataset(rows["train"], model.config, processor, config=opts, base_path=rows["train"].base_path)
    valid_ds = VisionDataset(rows["valid"], model.config, processor, config=opts, base_path=rows["valid"].base_path)
    image_token_id = model.config.image_token_id

    def batch_of(ds, i):
        return collate_vlm_batch([ds[i]], 4096, image_token_id=image_token_id)

    def loss_fn(model, batch):
        ids = batch["input_ids"]
        mask = batch["attention_mask"][:, 1:] * batch["completion_mask"][:, 1:]
        pos = mx.array(np.flatnonzero(np.array(mask[0])).astype(np.uint32))
        hidden = _forward_vlm(model, ids[:, :-1], batch["attention_mask"][:, :-1], batch).logits
        logits = project(hidden[0, pos]).astype(mx.float32)
        return nn.losses.cross_entropy(logits, ids[0, 1:][pos], reduction="mean"), pos.size

    steps_per_epoch = math.ceil(len(train_ds) / args.grad_accum)
    total = steps_per_epoch * args.epochs
    schedule = optim.join_schedules(
        [optim.linear_schedule(args.lr / 10, args.lr, args.warmup),
         optim.cosine_decay(args.lr, max(1, total - args.warmup), args.lr / 10)], [args.warmup])
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.0)
    value_and_grad = nn.value_and_grad(model, lambda m, b: loss_fn(m, b)[0])

    log = open(out / "train_log.csv", "a", newline="")
    lw = csv.writer(log)
    lw.writerow(["epoch", "opt_step", "samples", "train_loss", "answer_tokens", "lr", "seconds", "peak_memory_gb", "val_loss"])
    print(f"train {len(train_ds)}, valid {len(valid_ds)}, {total} optimizer steps", flush=True)

    seen, opt_step, t0 = 0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_ds)))
        random.shuffle(order)
        acc, acc_n, acc_loss, acc_tok = None, 0, 0.0, 0
        for k, i in enumerate(order, 1):
            batch = batch_of(train_ds, i)
            (loss, grads) = value_and_grad(model, batch)
            acc = grads if acc is None else tree_map(mx.add, acc, grads)
            mx.eval(loss, acc)
            acc_n += 1
            acc_loss += loss.item()
            acc_tok += int(batch["completion_mask"].sum().item())
            seen += 1
            if acc_n == args.grad_accum or k == len(order):
                optimizer.update(model, tree_map(lambda g: g / acc_n, acc))
                mx.eval(model.trainable_parameters(), optimizer.state)
                opt_step += 1
                lr = schedule(opt_step).item() if callable(schedule) else args.lr
                lw.writerow([epoch, opt_step, seen, f"{acc_loss / acc_n:.4f}", acc_tok, f"{lr:.2e}",
                             f"{time.time() - t0:.0f}", f"{mx.get_peak_memory() / 1e9:.2f}", ""])
                log.flush()
                print(f"epoch {epoch} step {opt_step}/{total} loss {acc_loss / acc_n:.4f} "
                      f"elapsed {time.time() - t0:.0f}s peak {mx.get_peak_memory() / 1e9:.1f}GB", flush=True)
                acc, acc_n, acc_loss, acc_tok = None, 0, 0.0, 0
                mx.clear_cache()
                if opt_step % 25 == 0:  # safety copy in case the run stops
                    save_adapter(model, out / "last" / "adapters.safetensors")
                    (out / "last" / "position.json").write_text(json.dumps({"epoch": epoch, "opt_step": opt_step, "samples": seen}))
            if args.max_steps and seen >= args.max_steps:
                break
        if args.max_steps and seen >= args.max_steps:
            save_adapter(model, out / "adapters.safetensors")
            return

        model.eval()
        vl = [loss_fn(model, batch_of(valid_ds, j))[0].item() for j in range(len(valid_ds))]
        model.train()
        val = sum(vl) / len(vl)
        lw.writerow([epoch, opt_step, seen, "", "", "", f"{time.time() - t0:.0f}", "", f"{val:.4f}"])
        log.flush()
        print(f"epoch {epoch} done: val_loss {val:.4f}", flush=True)
        save_adapter(model, out / f"epoch{epoch}" / "adapters.safetensors")
    save_adapter(model, out / "adapters.safetensors")
    print("saved", out / "adapters.safetensors", flush=True)


if __name__ == "__main__":
    main()
