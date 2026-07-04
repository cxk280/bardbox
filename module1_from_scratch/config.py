"""
Model + training hyper-parameters for the from-scratch GPT.

Everything here is sized to train on a CPU-only laptop (8 GB RAM, no GPU) in
minutes, not days. These are deliberately *small* numbers: this module exists to
prove understanding of the training loop, not to produce a useful model.

Two presets are provided:
  - SMOKE:  tiny + a handful of steps. Used to verify the whole pipeline runs
            end-to-end in ~a minute. This is the default so `python train.py`
            never hangs on a fresh checkout.
  - LAPTOP: the "real" tiny model you'd actually train for the report — still
            CPU-friendly, converges to readable (if nonsensical) Shakespeare.

Pick one by setting PRESET below, or override any field from the CLI in train.py.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    # --- model shape ---
    block_size: int = 128     # context length (chars of history the model sees)
    n_layer: int = 4          # number of transformer blocks stacked
    n_head: int = 4           # attention heads per block (n_embd must divide by this)
    n_embd: int = 128         # embedding / residual-stream width
    dropout: float = 0.1      # regularisation; small model + small data => keep some
    # vocab_size is set at runtime from the data (char-level), see train.py.

    # --- training ---
    batch_size: int = 32      # sequences per step; lower this if you hit RAM limits
    max_iters: int = 3000     # total optimisation steps
    eval_interval: int = 250  # how often to measure val loss + log
    eval_iters: int = 50      # batches averaged per loss estimate (less noisy)
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0    # clip global grad norm to stabilise training
    warmup_iters: int = 100   # linear LR warmup, then cosine decay


# A near-instant preset so the pipeline is verifiable on any machine in ~1 min.
SMOKE = GPTConfig(
    block_size=64, n_layer=2, n_head=2, n_embd=64,
    batch_size=16, max_iters=200, eval_interval=50, eval_iters=20, warmup_iters=20,
)

# The tiny-but-real preset used to generate report.md.
LAPTOP = GPTConfig()

# Default the CLI uses when no --preset is given. Keep SMOKE so a fresh clone
# `python train.py` finishes fast and proves the loop works before you commit
# to a longer run with `--preset laptop`.
PRESET = SMOKE
