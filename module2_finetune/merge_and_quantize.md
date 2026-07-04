# Merge the LoRA adapter and quantize to GGUF

After `colab_finetune.ipynb` finishes you have a **LoRA adapter** (a few MB of weight
deltas) sitting on top of the base model. To run it on the laptop in Act III we:

1. **merge** the adapter into the base weights → a standalone Hugging Face model, then
2. **convert** that to **GGUF** (the format `llama.cpp` loads), then
3. **quantize** it to 4-bit (`Q4_K_M`) so a 0.5B model is ~350 MB and runs fast on CPU.

Steps 1 is easiest to run at the end of the Colab notebook (it already has the model in
memory + a GPU). Steps 2–3 run locally on the Mac against a `llama.cpp` checkout.

---

## 1. Merge the adapter (in Colab, at the end of the notebook)

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# "bardbox-qwen-lora" is the adapter dir saved by the notebook.
model = AutoPeftModelForCausalLM.from_pretrained("bardbox-qwen-lora")
model = model.merge_and_unload()                 # fold LoRA deltas into base weights
model.save_pretrained("bardbox-qwen-merged")
AutoTokenizer.from_pretrained("bardbox-qwen-lora").save_pretrained("bardbox-qwen-merged")
```

Download `bardbox-qwen-merged/` (or push it to the Hugging Face Hub and pull it locally).

## 2. Convert to GGUF (locally, on the Mac)

```bash
# One-time: get llama.cpp and its Python conversion deps.
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt

# fp16 GGUF (full precision, large) — the intermediate we quantize from.
python convert_hf_to_gguf.py /path/to/bardbox-qwen-merged \
    --outfile bardbox-qwen-f16.gguf --outtype f16
```

## 3. Quantize to 4-bit (locally)

```bash
# Build llama.cpp's tools (CPU build is fine on an Intel Mac).
cmake -B build && cmake --build build --config Release -j

# Q4_K_M = 4-bit, the sweet spot of size vs quality for a small model on CPU.
./build/bin/llama-quantize bardbox-qwen-f16.gguf bardbox-qwen-q4.gguf Q4_K_M
```

You now have `bardbox-qwen-q4.gguf` (~350 MB for a 0.5B model). Sanity-check it:

```bash
./build/bin/llama-cli -m bardbox-qwen-q4.gguf -p "Rewrite in Shakespearean English: I am hungry." -n 60
```

This GGUF is the artifact Act III's app loads. Keep it out of git (it's large and
regenerable) — it's already covered by `.gitignore` (`*.gguf`).

---

### Why Q4_K_M and not smaller?

- `Q4_K_M` keeps quality very close to fp16 while cutting size ~4x — the standard default.
- On 8 GB RAM a 0.5–1.5B Q4 model leaves plenty of headroom for the OS + the app.
- Going to `Q2`/`Q3` saves little on a model this small and noticeably hurts the poetry.
