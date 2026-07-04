"""
Inference backends for the Writing Studio — pluggable, and every one of them local.

The privacy guarantee is architectural: there is no backend here that talks to a remote
host. The three implementations, in order of preference:

  1. LlamaCppPythonBackend — loads a GGUF model file in-process (real LLM, no network).
  2. LlamaServerBackend     — talks to a llama.cpp/Ollama server on 127.0.0.1 (local only).
  3. StubBackend            — a deterministic rule-based transformer; NO model required.

`select_backend()` is the composition root: it picks the best available backend at startup.
The Stub always works, so the app (and the no-egress test) are runnable on a fresh clone with
nothing installed. Drop a `.gguf` into module3_app/models/ (see module2_finetune/
merge_and_quantize.md) and the real model takes over automatically.

Every backend implements the same tiny interface (interface-segregation: one method).
"""

from __future__ import annotations

import glob
import os
import re
from typing import Protocol

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(HERE), "models")


class InferenceBackend(Protocol):
    name: str
    is_real_model: bool

    def generate(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        ...


# --------------------------------------------------------------------------------------
# 1. In-process GGUF via llama-cpp-python (real model, runs on CPU, no network)
# --------------------------------------------------------------------------------------
class LlamaCppPythonBackend:
    name = "llama-cpp-python (in-process GGUF)"
    is_real_model = True

    def __init__(self, model_path: str):
        from llama_cpp import Llama  # imported lazily so the dep is optional

        self.model_path = model_path
        # n_ctx modest to keep memory low on an 8 GB laptop; CPU-only by default.
        self._llm = Llama(model_path=model_path, n_ctx=2048, n_threads=os.cpu_count(), verbose=False)

    def generate(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        out = self._llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return out["choices"][0]["message"]["content"].strip()


# --------------------------------------------------------------------------------------
# 2. Local llama.cpp / Ollama server on 127.0.0.1 (OpenAI-compatible chat API)
# --------------------------------------------------------------------------------------
class LlamaServerBackend:
    name = "local llama.cpp/Ollama server"
    is_real_model = True

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def detect() -> "LlamaServerBackend | None":
        """Return a backend if a known local inference server answers, else None."""
        import httpx

        # llama.cpp server (8080) and Ollama (11434) both expose /v1/chat/completions.
        for base in ("http://127.0.0.1:8080", "http://127.0.0.1:11434"):
            try:
                r = httpx.get(base + "/v1/models", timeout=0.4)
                if r.status_code < 500:
                    return LlamaServerBackend(base)
            except Exception:
                continue
        return None

    def generate(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        import httpx

        r = httpx.post(
            self.base_url + "/v1/chat/completions",
            json={
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


# --------------------------------------------------------------------------------------
# 3. Deterministic stub — demo mode, no model. Rule-based modern -> Elizabethan.
# --------------------------------------------------------------------------------------
# Ordered phrase/word substitutions. Longer phrases first so they win over single words.
_PHRASES = [
    (r"\byou are\b", "thou art"), (r"\byou're\b", "thou art"),
    (r"\bdon't\b", "do not"), (r"\bdoesn't\b", "doth not"), (r"\bcan't\b", "cannot"),
    (r"\bisn't\b", "is not"), (r"\bwon't\b", "will not"), (r"\bi'm\b", "I am"),
    (r"\bi want\b", "I would"), (r"\bi need\b", "I have need of"),
    (r"\bgoing to\b", "about to"), (r"\bright now\b", "this very moment"),
]
_WORDS = {
    "you": "thou", "your": "thy", "yours": "thine", "yourself": "thyself",
    "are": "art", "have": "hast", "has": "hath", "do": "dost", "does": "doth",
    "please": "I prithee", "stop": "cease", "very": "most", "really": "truly",
    "yes": "aye", "no": "nay", "my": "mine", "before": "ere", "over": "o'er",
    "between": "'twixt", "here": "hither", "there": "thither", "where": "whither",
    "why": "wherefore", "beautiful": "most fair", "tired": "weary", "home": "hearth",
    "lying": "false", "lie": "falsehood", "hello": "well met", "goodbye": "fare thee well",
    "friend": "good fellow", "money": "coin", "happy": "full of joy", "sad": "sorrowful",
    "now": "anon", "often": "oft", "enough": "enow", "truly": "in sooth",
}
_OPENERS = ["", "Hark— ", "In sooth, ", "Prithee, ", "Marry, "]


def _elizabethanize(text: str, intensity: float) -> str:
    """A deterministic, honest caricature of Elizabethan English (NOT a language model)."""
    out = text
    for pat, repl in _PHRASES:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)

    def swap(m):
        w = m.group(0)
        low = w.lower()
        if low not in _WORDS:
            return w
        rep = _WORDS[low]
        return rep.capitalize() if w[0].isupper() else rep

    out = re.sub(r"[A-Za-z']+", swap, out)
    # At higher intensity, add a period-appropriate opener keyed deterministically to the text.
    if intensity >= 0.34:
        opener = _OPENERS[len(text) % len(_OPENERS)]
        if opener and not out.startswith(opener):
            out = opener + out[0].lower() + out[1:]
    out = re.sub(r"\bi\b", "I", out)  # never leave the first-person pronoun lowercased
    return out


class StubBackend:
    name = "demo stub (rule-based, no model)"
    is_real_model = False

    def generate(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        intensity = max(0.0, min(1.0, (temperature - 0.45) / 0.45)) if temperature else 0.5
        if "mentoring" in system:  # feedback task
            return (
                "'Tis a fair start, and thy meaning rings clear. Yet thou crowdest many thoughts "
                "into little space, and so each robs the others of their force. Choose the deepest "
                "and dwell there; let the rest fall silent. (Demo mode — add a GGUF for the real Bard.)"
            )
        if "conversing" in system:  # chat task
            return _elizabethanize(user, 0.7) + "  —Well, so I would answer, wert thou to ask it of the model itself."
        return _elizabethanize(user, intensity)  # rewrite task


# --------------------------------------------------------------------------------------
# Composition root — pick the best available backend once, at startup.
# --------------------------------------------------------------------------------------
def _find_gguf() -> str | None:
    hits = sorted(glob.glob(os.path.join(MODELS_DIR, "*.gguf")))
    return hits[0] if hits else None


def select_backend() -> InferenceBackend:
    """Prefer a real local model; fall back to the always-available stub."""
    gguf = _find_gguf()
    if gguf:
        try:
            return LlamaCppPythonBackend(gguf)
        except Exception:
            pass  # llama-cpp-python not installed — try a running server next
    server = LlamaServerBackend.detect()
    if server:
        return server
    return StubBackend()
