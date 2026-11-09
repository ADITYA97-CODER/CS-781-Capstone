"""Arch C: GPT-BERT-style hybrid encoder/decoder (one set of weights, two attention modes).

The same Transformer is trained on two kinds of batches (train.py, "mlm_prob"):
  * causal batches        causal attention, predict the next token      (like a GPT decoder)
  * masked batches        full bidirectional attention, predict tokens   (like a BERT encoder)
                          replaced by <mask> / random tokens
so it learns both left-to-right generation and two-sided context. For Phase-1
generation it runs in causal mode exactly like a decoder.

Default size (configs/archC.json): d=768, 20 layers, 12 heads x 64 dims, SwiGLU 2048 -> 166M params.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GenerationMixin, SwiGLU, apply_rope, init_weights, lm_loss, rope_tables


@dataclass
class ArchCConfig:
    arch: str = "C"
    vocab_size: int = 32000
    d_model: int = 768
    n_layers: int = 20
    n_heads: int = 12
    ffn_hidden: int = 2048
    max_seq_len: int = 1536
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


class SwitchableSelfAttention(nn.Module):
    """Self-attention whose mask is chosen per call: causal or fully bidirectional."""

    def __init__(self, cfg: ArchCConfig):
        super().__init__()
        self.n_heads, self.hd, self.dropout = cfg.n_heads, cfg.head_dim, cfg.dropout
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, bidirectional=False, kv_cache=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if kv_cache is not None:                  # generation is always causal
            k, v = kv_cache.update(k, v)
            causal = T > 1
        else:
            causal = not bidirectional
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


class Block(nn.Module):
    def __init__(self, cfg: ArchCConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = SwitchableSelfAttention(cfg)
        self.ffn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, bidirectional=False, kv_cache=None):
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin, bidirectional, kv_cache))
        return x + self.drop(self.ffn(self.ffn_norm(x)))


class ArchCHybrid(GenerationMixin, nn.Module):
    def __init__(self, cfg: ArchCConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        init_weights(self, cfg.n_layers)

    def forward(self, idx, targets=None, bidirectional=False):
        """bidirectional=True for masked batches: targets are the original tokens at masked
        positions and -100 elsewhere (ignored by the loss)."""
        T = idx.size(1)
        x = self.embed(idx)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin, bidirectional)
        logits = self.lm_head(self.norm(x))
        return logits, (lm_loss(logits, targets) if targets is not None else None)
