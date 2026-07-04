"""
A small decoder-only GPT, written from scratch and heavily commented.

This is the "I understand how an LLM works under the hood" artifact. It's the
same family of architecture as GPT-2 / Llama (token + position embeddings ->
stacked transformer blocks -> a linear head over the vocab), just tiny.

The forward pass is the whole story:
    ids ---embed---> vectors ---N transformer blocks---> vectors ---head---> logits

Each transformer block does two things, each wrapped in a residual connection:
    1. Causal multi-head self-attention  (tokens mix information with earlier tokens)
    2. A position-wise MLP               (each token is transformed independently)

"Causal" = a token may only attend to itself and tokens to its left, never the
future. That masking is what makes this a *language model* (predict the next token).
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (look-left-only) mask."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        # One linear layer produces query, key and value for every head at once
        # (3 * n_embd outputs), then we split them apart. Cheaper than 3 layers.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)  # mixes heads back together
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # Lower-triangular matrix of ones = the causal mask. Registered as a buffer
        # so it moves with the model but is not a learned parameter.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, channels (= n_embd)
        # Project to q, k, v and reshape each into (B, n_head, T, head_dim) so the
        # heads are an independent batch dimension attention runs over in parallel.
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Attention scores: how much each token should read from every other token.
        # Scale by 1/sqrt(head_dim) to keep the softmax from saturating.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        # Apply the causal mask: set future positions to -inf so softmax zeroes them.
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # weighted sum of value vectors -> (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-merge the heads
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network: expand 4x, GELU non-linearity, project back."""

    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """One transformer block: pre-norm attention + pre-norm MLP, both residual.

    "Pre-norm" (LayerNorm *before* each sub-layer) is what modern GPTs use — it
    makes deep stacks train stably. The `x = x + sublayer(norm(x))` shape is the
    residual connection that lets gradients flow straight through.
    """

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """The full model: embeddings -> blocks -> final norm -> vocab logits."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)      # id -> vector
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)        # position -> vector
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # vector -> logits
        # Weight tying: the input embedding and output projection share weights.
        # Standard trick that saves parameters and tends to improve quality.
        self.head.weight = self.token_emb.weight
        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialised with {n_params/1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        # Add token meaning and token position, then run the stack.
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy over the vocab at every position: "what's the next char?"
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively sample characters, one at a time, feeding each back in."""
        self.eval()
        for _ in range(max_new_tokens):
            # Never feed more than block_size chars of context (the model's limit).
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # only the last position matters
            if top_k is not None:  # optionally keep only the k most likely chars
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # sample, don't argmax
            idx = torch.cat((idx, next_id), dim=1)
        return idx
