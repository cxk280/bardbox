"""
Generate text from the trained from-scratch GPT.

Run (from inside module1_from_scratch/):
    python sample.py                          # 500 chars from a newline prompt
    python sample.py --prompt "ROMEO:" -n 300 # continue a prompt
    python sample.py --temperature 0.8 --top_k 40

Loads checkpoints/model.pt and data/meta.json (the char<->id map), samples one
character at a time, and decodes the ids back to text.
"""

import argparse
import json
import os

import torch

from config import GPTConfig
from model import GPT

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="\n", help="starting text to continue")
    ap.add_argument("-n", "--num_tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.8, help="lower = more conservative")
    ap.add_argument("--top_k", type=int, default=40, help="sample only from top-k chars")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # Rebuild the tokenizer from meta.json.
    with open(os.path.join(HERE, "data", "meta.json")) as f:
        meta = json.load(f)
    stoi = meta["stoi"]
    itos = {int(k): v for k, v in meta["itos"].items()}  # JSON keys are strings
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)

    # Rebuild the model from the saved config and load the trained weights.
    ckpt = torch.load(os.path.join(HERE, "checkpoints", "model.pt"), weights_only=False)
    cfg: GPTConfig = ckpt["config"]
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Encode the prompt, generate, decode.
    start_ids = encode(args.prompt)
    idx = torch.tensor([start_ids], dtype=torch.long)
    out = model.generate(
        idx, max_new_tokens=args.num_tokens,
        temperature=args.temperature, top_k=args.top_k,
    )
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
