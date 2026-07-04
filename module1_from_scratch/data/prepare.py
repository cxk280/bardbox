"""
Download the tiny-shakespeare corpus and build a character-level tokenizer.

Char-level = the "tokenizer" is just: sort the unique characters in the text and
assign each one an integer id. No BPE, no external vocab file — this keeps the
whole model self-contained and makes the training-from-scratch story easy to read.

Run:
    python data/prepare.py      # from inside module1_from_scratch/

Produces (next to this file):
    input.txt   - the raw corpus (~1.1 MB, downloaded once)
    meta.json   - the char<->id mapping + vocab_size, so sample.py can decode later
    train.bin / val.bin - the corpus encoded as uint16 ids, 90/10 split

The .bin/.txt files are git-ignored (regenerable), so a clean checkout just re-runs this.
"""

import json
import os
from urllib.request import urlretrieve

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(HERE, "input.txt")
# The canonical tiny-shakespeare file (Karpathy's char-rnn corpus): the concatenated
# works of Shakespeare, ~1.1 MB of plain text. Public domain.
DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def main():
    # 1. Fetch the corpus once (this is the ONLY network access in Module 1).
    if not os.path.exists(INPUT_PATH):
        print(f"Downloading tiny-shakespeare -> {INPUT_PATH}")
        urlretrieve(DATA_URL, INPUT_PATH)
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Corpus length: {len(text):,} characters")

    # 2. Build the char-level vocabulary: every distinct character, sorted for
    #    determinism, mapped to a contiguous integer id.
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}   # string -> int
    itos = {i: ch for i, ch in enumerate(chars)}   # int -> string
    print(f"Vocab size: {vocab_size} unique characters")

    # 3. Encode the whole corpus to ids and split 90/10 into train/val.
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)
    n = int(0.9 * len(ids))
    train_ids, val_ids = ids[:n], ids[n:]
    train_ids.tofile(os.path.join(HERE, "train.bin"))
    val_ids.tofile(os.path.join(HERE, "val.bin"))
    print(f"train.bin: {len(train_ids):,} tokens   val.bin: {len(val_ids):,} tokens")

    # 4. Persist the tokenizer so generation can decode ids back to text.
    with open(os.path.join(HERE, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab_size": vocab_size, "itos": itos, "stoi": stoi}, f)
    print("Wrote meta.json")


if __name__ == "__main__":
    main()
