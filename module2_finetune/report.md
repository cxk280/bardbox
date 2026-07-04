# Act II report — fine-tuning a real model

## What this module demonstrates

Act I showed the *mechanics* of training but produced a useless model — a laptop can't
pretrain a capable LLM. Act II shows the technique that actually produces useful behavior
on a budget: **take a model someone else already pretrained, and cheaply specialize it.**

We fine-tune **Qwen2.5-0.5B-Instruct** — a small, ungated, open-source model that already
speaks fluent English — to a single skill: *rewrite modern English in Shakespeare's voice*.

## Why LoRA / QLoRA (and why it fits a free GPU)

Full fine-tuning updates all 0.5B weights — too much memory for a free tier. Instead:

- **LoRA** freezes the base model and trains small low-rank "adapter" matrices injected into
  the attention/MLP projections. Here that's a few million trainable params, not 0.5B.
- **QLoRA** additionally loads the frozen base in **4-bit** (`nf4`), cutting its memory ~4x.

Together they let a 0.5B fine-tune run comfortably on a **free Colab/Kaggle T4** in minutes.
The output is a tiny adapter (a few MB) you merge back into the base when done.

## The data

`build_dataset.py` turns modern↔Shakespearean sentence pairs into chat-formatted SFT examples
(`data/train.jsonl` / `val.jsonl`). Because the base model already knows English, we're only
teaching a **style shift**, so a small curated set goes a long way. Expand `data/pairs.tsv`
(`modern<TAB>shakespeare`) to grow it before a serious run.

## Two ways to run it (both free)

- **Recommended — free GPU:** open `colab_finetune.ipynb` in Google Colab, set runtime to **T4
  GPU**, provide `data/train.jsonl` / `val.jsonl`, run all cells. Uses QLoRA (4-bit) and finishes
  in a couple of minutes. It prints BEFORE/AFTER rewrites, then merges the adapter and saves
  `bardbox-qwen-merged/`.
- **No GPU at all — CPU:** `.venv/bin/python module2_finetune/local_finetune_cpu.py`. Same LoRA,
  in fp32 on the laptop CPU, with an explicit hand-written training loop. Slower (~25 min on a
  dual-core i5) but proves the fine-tune needs *no* special hardware. This is the run that
  produced the results below.

Then follow [`merge_and_quantize.md`](merge_and_quantize.md) to produce the Q4 GGUF that Act III
loads.

## Results — real CPU run (8.8M LoRA params, 1.75% of the model)

Training loss fell **2.38 → 0.05** over 8 epochs. Same three prompts, before vs after:

| Modern input | Base Qwen (before) | Fine-tuned (after) |
|--------------|--------------------|--------------------|
| "I am very tired and I want to go home." | *"O ye weary sirs, with weary feet upon the ground, where we seek thy rest? Here, my lords and ladies, I am weary of thee, for thou dost not seem fit for thy journey home. Yet, if it be true…"* (rambles, drifts off the meaning, hits the length cap) | *"My heart is heavy upon mine feet, and I pray thee, wouldst thou grant it me one further journey."* |
| "Please stop lying to me." | *"Behold, ye who speak with lies; for truth is but a shadow upon this false world of deceit. Let truth and sincerity reign supreme within your hearts and minds."* | *"Let no lie remain upon thy tongue."* |
| "The weather is beautiful today." | *"There was an ambiance of delight and bliss that permeated every corner as the sun shone brightly upon our scene, illuminating all the elements with its radiance…"* | *"The heavens smile most fair upon this day."* |

### What the fine-tune actually changed (honest reading)

- **The base model was already flowery** — a nice surprise, and worth stating plainly. Qwen2.5-0.5B
  can *do* archaic English unprompted. So this isn't a "gibberish → poetry" story.
- **What LoRA added is control, not vocabulary:** the fine-tuned model is **concise, faithful to the
  original meaning, and knows when to stop.** The base model free-associates and blows past the
  point; the tuned model delivers one clean epigram. That "learn the task's *shape*, not just its
  words" is exactly what instruction fine-tuning buys you.
- **One output is memorized:** "The heavens smile most fair upon this day" is verbatim a training
  pair — expected mild overfitting on only 45 examples. The other two outputs are novel (not in the
  data), so the adapter generalized the *style*, it didn't just memorize. Growing `data/pairs.tsv`
  to a few hundred pairs would reduce the memorization.

These outputs feed the three-way comparison in
[`../docs/comparison.md`](../docs/comparison.md) (from-scratch vs base vs fine-tuned).
