"""
Train the from-scratch GPT on tiny-shakespeare (CPU-only).

This is the training loop laid out plainly so it reads as a demonstration:
    get a batch -> forward (compute loss) -> backward (gradients) -> optimiser step
    ...every eval_interval steps, measure train+val loss and log it to CSV.

Run (from inside module1_from_scratch/):
    python data/prepare.py          # once, to create train.bin / val.bin / meta.json
    python train.py                 # fast SMOKE preset (verifies the loop)
    python train.py --preset laptop # the tiny-but-real run used for report.md

Outputs:
    checkpoints/model.pt   - trained weights + config (consumed by sample.py)
    loss_log.csv           - step, train_loss, val_loss (consumed by plot in report)
"""

import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch

from config import LAPTOP, SMOKE
from model import GPT

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CKPT_DIR = os.path.join(HERE, "checkpoints")


def get_batch(data, cfg, device):
    """Grab batch_size random (input, target) windows of length block_size.

    target is input shifted right by one char: at each position the model learns
    to predict the *next* character. This is the whole training signal.
    """
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + cfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, cfg, device):
    """Average loss over several batches per split — a less-noisy loss estimate."""
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(data, cfg, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_for_step(step, cfg):
    """Linear warmup then cosine decay to 10% of the base LR — the standard schedule."""
    if step < cfg.warmup_iters:
        return cfg.learning_rate * (step + 1) / cfg.warmup_iters
    progress = (step - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return cfg.learning_rate * (0.1 + 0.9 * coeff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["smoke", "laptop"], default="smoke")
    args = ap.parse_args()
    cfg = SMOKE if args.preset == "smoke" else LAPTOP

    torch.manual_seed(1337)  # reproducible runs
    device = "cpu"  # this whole module is intentionally CPU-only
    print(f"Preset: {args.preset}  |  device: {device}")

    # Load the encoded corpus + tokenizer metadata produced by data/prepare.py.
    meta_path = os.path.join(DATA_DIR, "meta.json")
    if not os.path.exists(meta_path):
        raise SystemExit("Run `python data/prepare.py` first to build the dataset.")
    with open(meta_path) as f:
        cfg.vocab_size = json.load(f)["vocab_size"]
    train_data = np.fromfile(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16)
    val_data = np.fromfile(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16)
    splits = {"train": train_data, "val": val_data}

    model = GPT(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    os.makedirs(CKPT_DIR, exist_ok=True)
    log_path = os.path.join(HERE, "loss_log.csv")
    log_file = open(log_path, "w", newline="")
    logger = csv.writer(log_file)
    logger.writerow(["step", "train_loss", "val_loss"])

    print("Starting training...")
    t0 = time.time()
    for step in range(cfg.max_iters + 1):
        # Apply the current point on the LR schedule.
        lr = lr_for_step(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Periodically measure + log loss (and at the very end).
        if step % cfg.eval_interval == 0 or step == cfg.max_iters:
            losses = estimate_loss(model, splits, cfg, device)
            dt = time.time() - t0
            print(
                f"step {step:>5} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | lr {lr:.2e} | {dt:.0f}s"
            )
            logger.writerow([step, f"{losses['train']:.4f}", f"{losses['val']:.4f}"])
            log_file.flush()

        if step == cfg.max_iters:
            break

        # --- the core optimisation step ---
        x, y = get_batch(train_data, cfg, device)
        _, loss = model(x, y)                       # forward: predictions + loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()                             # backward: gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()                            # nudge weights downhill

    log_file.close()

    # Save weights + the config + vocab_size so sample.py can rebuild the model.
    ckpt_path = os.path.join(CKPT_DIR, "model.pt")
    torch.save({"model": model.state_dict(), "config": cfg}, ckpt_path)
    print(f"Saved checkpoint -> {ckpt_path}")
    print(f"Loss log -> {log_path}")


if __name__ == "__main__":
    main()
