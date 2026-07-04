"""
CPU-only LoRA fine-tune — the no-GPU-at-all fallback for Act II.

colab_finetune.ipynb is the recommended path (a free T4 is faster and uses QLoRA 4-bit).
But the whole ethos of this repo is "runs on a laptop with no GPU," so this script proves
the *same* fine-tune works on plain CPU too — just slower and in fp32 (no bitsandbytes).

It uses a small, hand-written training loop (rather than trl's SFTTrainer) so it's robust to
library version churn and reads as an explicit demonstration of what fine-tuning actually does.

Run (from repo root, in the venv, after `pip install transformers peft`):
    .venv/bin/python module2_finetune/local_finetune_cpu.py

Outputs:
    module2_finetune/bardbox-qwen-lora/   - the trained LoRA adapter
    module2_finetune/results.json         - before/after rewrites (feeds report + comparison)
"""

import json
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
BASE = "Qwen/Qwen2.5-0.5B-Instruct"  # ungated, tiny, has a chat template

# Prompts we evaluate before AND after training, to show the fine-tune's effect.
EVAL_PROMPTS = [
    "I am very tired and I want to go home.",
    "Please stop lying to me.",
    "The weather is beautiful today.",
]


def build_chat(tok, messages, add_generation_prompt):
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def rewrite(model, tok, text, max_new_tokens=60):
    """Ask the model to rewrite one sentence in Shakespeare's voice."""
    msgs = [
        {"role": "system", "content": "You are a Shakespearean playwright. Rewrite the user's modern English in Shakespeare's style."},
        {"role": "user", "content": text},
    ]
    prompt = build_chat(tok, msgs, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


def load_training_examples(tok):
    """Turn each chat example into a (input_ids, labels) pair.

    labels mask the prompt tokens with -100 so the loss is computed ONLY on the
    assistant's Shakespearean reply — the model is scored on what it should generate,
    not on echoing the instruction. This label-masking is the crux of instruction SFT.
    """
    examples = []
    with open(os.path.join(DATA_DIR, "train.jsonl"), encoding="utf-8") as f:
        for line in f:
            msgs = json.loads(line)["messages"]
            # Full conversation (prompt + answer) and just the prompt portion.
            full = build_chat(tok, msgs, add_generation_prompt=False)
            prompt_only = build_chat(tok, msgs[:-1], add_generation_prompt=True)
            full_ids = tok(full, return_tensors="pt").input_ids[0]
            prompt_len = tok(prompt_only, return_tensors="pt").input_ids.shape[1]
            labels = full_ids.clone()
            labels[:prompt_len] = -100  # ignore prompt tokens in the loss
            examples.append((full_ids, labels))
    return examples


def main():
    torch.manual_seed(1337)
    print(f"Loading {BASE} on CPU (fp32)... first run downloads ~1 GB.")
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
    model.train()

    # BEFORE: capture the base model's rewrites so we can compare after training.
    print("\n=== BEFORE fine-tuning ===")
    before = {p: rewrite(model, tok, p) for p in EVAL_PROMPTS}
    for p, r in before.items():
        print(f"  {p}\n   -> {r}\n")

    # Attach a LoRA adapter: freeze the 0.5B base, train small low-rank matrices.
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    examples = load_training_examples(tok)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4
    )

    EPOCHS = 8
    print(f"\nTraining {len(examples)} examples for {EPOCHS} epochs on CPU...")
    for epoch in range(EPOCHS):
        total = 0.0
        for input_ids, labels in examples:
            out = model(input_ids=input_ids.unsqueeze(0), labels=labels.unsqueeze(0))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total += out.loss.item()
        print(f"  epoch {epoch+1}/{EPOCHS} | mean loss {total/len(examples):.4f}")

    # AFTER: same prompts, now through the fine-tuned model.
    model.eval()
    print("\n=== AFTER fine-tuning ===")
    after = {p: rewrite(model, tok, p) for p in EVAL_PROMPTS}
    for p, r in after.items():
        print(f"  {p}\n   -> {r}\n")

    model.save_pretrained(os.path.join(HERE, "bardbox-qwen-lora"))
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"base_model": BASE, "before": before, "after": after}, f, indent=2, ensure_ascii=False)
    print("Saved adapter -> bardbox-qwen-lora/  and results -> results.json")


if __name__ == "__main__":
    main()
