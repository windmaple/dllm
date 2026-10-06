"""MiniGPT: a shrunk-down GPT-2 written from scratch in PyTorch.

The only architectural change needed to turn GPT-2 into a *masked diffusion*
language model is to drop the causal mask: the denoiser looks at the whole
(partially masked) sequence at once and predicts the clean token at every
masked position.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MiniGPTConfig:
    vocab_size: int = 8192
    block_size: int = 256  # max sequence length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True  # GPT-2 uses biases in Linear and LayerNorm layers


class SelfAttention(nn.Module):
    """Multi-head self-attention (bidirectional, i.e. no causal mask)."""

    def __init__(self, cfg: MiniGPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # <-- the key difference from autoregressive GPT-2
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: MiniGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    """Pre-LayerNorm transformer block, as in GPT-2."""

    def __init__(self, cfg: MiniGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = SelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: MiniGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # token embeddings
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)  # learned positions
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying, as in GPT-2

        # GPT-2 style init
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()
        return n

    def forward(self, idx, only=None):
        """idx: (B, T) token ids (may contain the <mask> id).

        Returns logits (B, T, V), or (N, V) for the N positions where the
        boolean tensor `only` is True (skips the LM head on unmasked tokens).
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        if only is not None:
            x = x[only]
        return self.lm_head(x)
