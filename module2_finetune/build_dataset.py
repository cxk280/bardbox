"""
Build the Shakespeare style-transfer fine-tuning dataset.

Act II teaches a *real* model an instruction: "rewrite this modern sentence in
Shakespearean English." To fine-tune for that we need (instruction, input, output)
examples. We generate them from parallel modern<->Elizabethan sentence pairs kept in
`pairs.tsv`, formatted as chat messages the way an instruct model expects.

This runs anywhere (pure Python, no ML deps) so the dataset is reproducible without a GPU.
The actual fine-tuning happens in colab_finetune.ipynb on a free GPU.

Run (from inside module2_finetune/):
    python build_dataset.py

Outputs:
    data/train.jsonl / data/val.jsonl  - chat-formatted SFT examples (90/10 split)

Each line is: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
which is the format trl's SFTTrainer + a chat template consume directly.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

SYSTEM_PROMPT = (
    "You are a Shakespearean playwright. Rewrite the user's modern English text in the "
    "elevated, poetic style of William Shakespeare, preserving the original meaning."
)

# A small, hand-curated seed set of modern -> Shakespearean pairs. Small is fine:
# LoRA on top of a pretrained model that already speaks English needs to learn the
# *style shift*, not the language, so a few hundred good examples go a long way.
# In a real project you'd expand pairs.tsv to ~1–3k lines; this seed proves the pipeline.
SEED_PAIRS = [
    ("Where are you going?", "Whither dost thou wend thy way?"),
    ("I am very tired.", "In faith, a heavy weariness doth weigh upon me."),
    ("Please leave me alone.", "I prithee, grant me leave to be alone."),
    ("You are lying to me.", "Thou speak'st me false, and well thou know'st it."),
    ("I love you more than anything.", "I love thee past all measure of the world."),
    ("This is a terrible idea.", "This counsel is most rank and ill-conceived."),
    ("What do you want from me?", "What wouldst thou have of me, I pray thee tell?"),
    ("He betrayed his best friend.", "He hath betray'd the friend that loved him dearest."),
    ("I can't believe you did that.", "Scarce can I credit that thy hand hath wrought this deed."),
    ("Let's meet tomorrow morning.", "Come, let us meet upon the morrow's early light."),
    ("I'm afraid of what comes next.", "A dread doth grip me for what fate may bring."),
    ("Stop wasting my time.", "Waste not the precious hours that are mine."),
    ("She is the smartest person I know.", "No wiser soul doth walk beneath the sun than she."),
    ("Money can't buy happiness.", "No coin nor coffer purchaseth true joy."),
    ("Tell me the truth.", "Unfold to me the very truth, withhold it not."),
    ("I made a terrible mistake.", "Alas, mine own hand hath wrought a grievous error."),
    ("The weather is beautiful today.", "The heavens smile most fair upon this day."),
    ("I will never forgive you.", "Ne'er shall my heart grant thee its pardon more."),
    ("We need to talk.", "There is a matter 'twixt us that must be spoke."),
    ("Everything is going to be fine.", "Fear not; all shall be well, I warrant thee."),
]


def to_example(modern: str, bard: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": modern},
            {"role": "assistant", "content": bard},
        ]
    }


def load_pairs() -> list[tuple[str, str]]:
    """Prefer data/pairs.tsv if present (so you can expand the set without editing code);
    otherwise fall back to the SEED_PAIRS baked in above."""
    tsv = os.path.join(DATA_DIR, "pairs.tsv")
    if os.path.exists(tsv):
        pairs = []
        with open(tsv, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or "\t" not in line:
                    continue
                modern, bard = line.split("\t", 1)
                pairs.append((modern, bard))
        return pairs
    return SEED_PAIRS


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    pairs = load_pairs()
    examples = [to_example(m, b) for m, b in pairs]

    # 90/10 train/val split. Deterministic (no shuffle) so the dataset is reproducible.
    n_val = max(1, len(examples) // 10)
    val, train = examples[:n_val], examples[n_val:]

    for name, rows in [("train", train), ("val", val)]:
        path = os.path.join(DATA_DIR, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {path}: {len(rows)} examples")

    print(
        f"\nTotal {len(examples)} examples. Expand data/pairs.tsv (modern<TAB>shakespeare) "
        "to grow the set before a real fine-tune."
    )


if __name__ == "__main__":
    main()
