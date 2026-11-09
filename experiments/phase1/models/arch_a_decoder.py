"""Arch A: Llama-style decoder-only Transformer (the main Phase-1 model).

Every layer uses full causal self-attention: token t attends to tokens 0..t.
Components: RMSNorm pre-normalization, RoPE, SwiGLU, no biases, tied embeddings.

Default size (configs/archA.json): d=1024, 16 layers, 16 heads x 64 dims, SwiGLU 2816 -> 238M params.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GenerationMixin, SwiGLU, apply_rope, init_weights, lm_loss, rope_tables


@dataclass
class ArchAConfig:
    arch: str = "A"
    vocab_size: int = 32000
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    ffn_hidden: int = 2816
    max_seq_len: int = 1536
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ArchAConfig):
        super().__init__()
        self.n_heads, self.hd, self.dropout = cfg.n_heads, cfg.head_dim, cfg.dropout
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, kv_cache=None):
        B, T, _ = x.shape
        # (B, T, d) -> (B, heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)
        # Training and prompt prefill: causal mask (Flash Attention kernel).
        # Single-token decode step: the new token may attend to every cached key.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


class Block(nn.Module):
    """Pre-norm residual block: x + Attn(Norm(x)), then x + FFN(Norm(x))."""

    def __init__(self, cfg: ArchAConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, kv_cache=None):
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin, kv_cache))
        return x + self.drop(self.ffn(self.ffn_norm(x)))


class ArchADecoder(GenerationMixin, nn.Module):
    def __init__(self, cfg: ArchAConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight          # tied input/output embeddings
        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        init_weights(self, cfg.n_layers)

    def forward(self, idx, targets=None):
        T = idx.size(1)
        x = self.embed(idx)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.norm(x))
        return logits, (lm_loss(logits, targets) if targets is not None else None)
