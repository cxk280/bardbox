# The three-way comparison — why each technique earns its place

This is the payoff of the whole repo. The **same task** — "write like Shakespeare" — run through
the three models the three modules produce, so you can see capability climb as we move from
laptop-pretraining to borrowing-a-pretrained-model to specializing-it.

| Model | How it was made | Runs on |
|-------|-----------------|---------|
| **A. From-scratch GPT** (Act I) | 0.82M params, char-level, pretrained on tiny-shakespeare, on a laptop CPU (val loss 1.88) | Laptop CPU |
| **B. Base Qwen2.5-0.5B-Instruct** (Act II input) | Pretrained by Alibaba on trillions of tokens; we did nothing to it | Laptop CPU / free GPU |
| **C. Fine-tuned Qwen** (Act II output) | Model B + a LoRA adapter trained on our Shakespeare style set | Laptop CPU / free GPU |

## Free-generation sample (no instruction — just "continue")

Model A can only *continue* text (it has no notion of instructions). Prompt: `ROMEO:`

```
ROMEO:
Be bother have can mise bralies us pasteed.

QUEEN VINCENTIO:
In pecess.

RUCETIO:
And that mast bradyer the of homise an the you call.
```

**Read it for:** correct Shakespearean *shape* (character names, line breaks, archaic words) but
no coherent meaning. That's an honest 0.82M-param char model trained ~37 min on one CPU.

## Instruction task: "Rewrite in Shakespearean English"

Models B and C can follow an instruction; Model A cannot (it was never trained to), which is
itself an instructive result — capability comes from *pretraining scale + instruction tuning*,
not from the tiny local model.

| Modern input | A. From-scratch | B. Base Qwen | C. Fine-tuned Qwen |
|--------------|-----------------|--------------|--------------------|
| "I am very tired and I want to go home." | *cannot follow instructions — emits Shakespeare-shaped noise* | *"O ye weary sirs, with weary feet upon the ground, where we seek thy rest?… I am weary of thee, for thou dost not seem fit for thy journey home. Yet, if it be true…"* (rambles past the point) | *"My heart is heavy upon mine feet, and I pray thee, wouldst thou grant it me one further journey."* |
| "Please stop lying to me." | *n/a* | *"Behold, ye who speak with lies; for truth is but a shadow upon this false world of deceit. Let truth and sincerity reign supreme within your hearts and minds."* | *"Let no lie remain upon thy tongue."* |
| "The weather is beautiful today." | *n/a* | *"There was an ambiance of delight and bliss that permeated every corner as the sun shone brightly… like the breath of a summer breeze…"* | *"The heavens smile most fair upon this day."* |

## The takeaway (the portfolio thesis)

- **A → B:** you cannot brute-force capability on a laptop. Model A can't even *follow an
  instruction* — it was never pretrained at the scale that produces that ability. Scale of
  pretraining is what makes a model *useful*; on a budget you stand on a pretrained open model.
- **B → C:** the surprise here is that base Qwen-0.5B is *already* flowery, so the win from a cheap
  LoRA fine-tune isn't vocabulary — it's **control**. C is concise, stays faithful to the input's
  meaning, and knows when to stop, where B free-associates and overruns. Fine-tuning teaches the
  *shape of the task*, and it's the highest-leverage, lowest-cost customization you can do (here:
  8.8M adapter params, ~1.75% of the model, trained in minutes on a **CPU**).
- **All three run on the same cheap hardware**, and **C runs fully offline** — which is what makes
  Act III's privacy guarantee possible.

> Reproduce these exact columns: A via `module1_from_scratch/sample.py`; B and C via
> `module2_finetune/local_finetune_cpu.py` (writes `module2_finetune/results.json`).
