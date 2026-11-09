"""Arch E: hybrid local/global decoder with differential attention and tied embeddings.

  * Layer pattern (`layer_pattern`): "alternate" = local, global, local, ... with the last layer
    global; "all_global"; "every_k:<k>" (every k-th layer global, like Arch D); or an explicit
    list such as ["L", "G", ...]. Local layers attend to the last `window` tokens.
  * Differential attention (DIFF Transformer; `diff_layers` = "all" | "global_only" | "none"):
    query/key heads are split in two halves of width head_dim, value heads are 2*head_dim wide, so
    there are half as many heads and the Q/K/V/O shapes equal standard attention.
        A1 = attn(q1, k1, v),  A2 = attn(q2, k2, v)           (same mask, scale 1/sqrt(head_dim))
        out = RMSNorm_per_head(A1 - lam * A2) * (1 - lam_init)
        lam = exp(lq1 . lk1) - exp(lq2 . lk2) + lam_init,  lam_init = 0.8 - 0.6 exp(-0.3 (l - 1))
    The two maps cancel common-mode attention noise. Implemented with FlexAttention; each map is
    applied to the two head_dim halves of v separately (4 calls per layer, see `attend`).
  * Grouped-query attention, QK-norm (RMSNorm per head before RoPE), SwiGLU, pre-norm RMSNorm,
    no biases.
  * doc_mask (default on): attention stays inside the current document (plus the window in
    local layers) and RoPE positions restart at every document (as in Arch D).
  * tie_embeddings: lm_head.weight is the embedding matrix (one tensor). The vocabulary is padded
    to a multiple of `vocab_pad_to`; logits of the padding rows are dropped.

Default size (configs/archE.json): d=768, 20 layers, 12 query / 4 KV heads (6 / 2 differential
heads), head_dim 64, SwiGLU 1920, tied -> 144.56M parameters.
"""

import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from .arch_d_sliding import ArchDSliding
from .layers import GenerationMixin, SwiGLU, apply_rope, doc_structure, init_weights, lm_loss, rope_tables


@dataclass
class ArchEConfig:
    arch: str = "E"
    vocab_size: int = 32000
    vocab_pad_to: int = 128
    d_model: int = 768
    n_layers: int = 20
    n_heads: int = 12             # standard-attention query heads (halved for differential attention)
    n_kv_heads: int = 4
    head_dim: int = 64
    ffn_hidden: int = 1920
    max_seq_len: int = 1536
    window: int = 512
    layer_pattern: Union[str, list] = "alternate"
    diff_layers: str = "all"      # "all" | "global_only" | "none"
    qk_norm: bool = True
    tie_embeddings: bool = True
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    doc_mask: bool = True
    eos_id: int = 0

    def __post_init__(self):
        assert self.n_heads % self.n_kv_heads == 0
        assert len(self.layer_kinds()) == self.n_layers
        assert self.diff_layers in ("all", "global_only", "none")
        if any(self.is_diff(i) for i in range(self.n_layers)):
            assert self.n_heads % 2 == 0 and self.n_kv_heads % 2 == 0, "differential attention halves the heads"
            assert (self.n_heads // 2) % (self.n_kv_heads // 2) == 0, \
                "differential query heads must be divisible by differential KV heads"

    @property
    def padded_vocab(self):
        return -(-self.vocab_size // self.vocab_pad_to) * self.vocab_pad_to

    def layer_kinds(self):
        """'L' (local) or 'G' (global) for every layer."""
        p, L = self.layer_pattern, self.n_layers
        if isinstance(p, (list, tuple)):
            return [str(k).upper()[0] for k in p]
        if p == "all_global":
            return ["G"] * L
        if p == "all_local":
            return ["L"] * L
        if p == "alternate":                     # counted from the end, so the last layer is global
            return ["G" if (L - 1 - i) % 2 == 0 else "L" for i in range(L)]
        if p.startswith("every_k:"):
            k = int(p.split(":")[1])
            return ["G" if (i + 1) % k == 0 else "L" for i in range(L)]
        raise ValueError(f"unknown layer_pattern {p!r}")

    def is_local(self, i):
        return self.layer_kinds()[i] == "L"

    def is_diff(self, i):
        return self.diff_layers == "all" or (self.diff_layers == "global_only" and not self.is_local(i))


def attend(q, k, v, mask, scale):
    """One attention map applied to v: FlexAttention with a BlockMask, or SDPA with a boolean mask
    (True = attend; generation and the dense fallback).
    Differential layers have value heads twice as wide as q/k. FlexAttention in torch 2.6 cannot
    take that on this GPU: natively, compiled no-grad calls infer the output width from q; with q/k
    zero-padded to the value width, the compiled backward has no valid kernel for head_dim 128 on
    sm_86. So v is split into two head_dim-wide halves (two calls with the same q, k and mask) and
    the outputs are concatenated, which is exact."""
    if isinstance(mask, BlockMask):
        if v.size(-1) != q.size(-1):
            return torch.cat([flex_attention(q, k, vh, block_mask=mask, scale=scale, enable_gqa=True)
                              for vh in v.split(q.size(-1), dim=-1)], dim=-1)
        return flex_attention(q, k, v, block_mask=mask, scale=scale, enable_gqa=True)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=True)


class Attention(nn.Module):
    """Grouped-query attention, differential or standard, local (sliding window) or global."""

    def __init__(self, cfg: ArchEConfig, layer_idx: int):
        super().__init__()
        self.local, self.diff, self.window = cfg.is_local(layer_idx), cfg.is_diff(layer_idx), cfg.window
        hd = self.hd = cfg.head_dim
        self.scale = 1.0 / math.sqrt(hd)
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * hd, cfg.d_model, bias=False)
        self.q_norm = nn.RMSNorm(hd, eps=cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(hd, eps=cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        if self.diff:
            self.n_heads, self.n_kv = cfg.n_heads // 2, cfg.n_kv_heads // 2     # heads of width 2*hd
            self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)          # layer_idx = l - 1
            self.lambda_q1 = nn.Parameter(torch.randn(hd) * 0.1)
            self.lambda_k1 = nn.Parameter(torch.randn(hd) * 0.1)
            self.lambda_q2 = nn.Parameter(torch.randn(hd) * 0.1)
            self.lambda_k2 = nn.Parameter(torch.randn(hd) * 0.1)
            self.subln = nn.Parameter(torch.ones(self.n_heads, 2 * hd))        # per-head RMSNorm weight
            self.norm_eps = cfg.norm_eps
        else:
            self.n_heads, self.n_kv = cfg.n_heads, cfg.n_kv_heads

    def lam(self):
        """Scalar lambda of this layer, in fp32."""
        a = torch.exp((self.lambda_q1.float() * self.lambda_k1.float()).sum())
        b = torch.exp((self.lambda_q2.float() * self.lambda_k2.float()).sum())
        return a - b + self.lambda_init

    def project(self, x, cos, sin):
        """-> q (B, n_q, T, hd), k (B, n_k, T, hd), v (B, n_kv, T, hd or 2*hd), normed and rotated.
        Differential layers: hd-wide heads 2i and 2i+1 are q1 and q2 (k1 and k2) of head i."""
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, -1, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, -1, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, -1).transpose(1, 2)
        # under autocast the QK-norm output is fp32: bring q and k back to v's dtype
        return apply_rope(q, cos, sin).type_as(v), apply_rope(k, cos, sin).type_as(v), v

    def combine(self, q, k, v, mask):
        """Attention over prepared heads with one mask -> (B, T, n_heads * width)."""
        B, _, T, _ = q.shape
        if not self.diff:
            out = attend(q, k, v, mask, self.scale)
        else:
            a1 = attend(q[:, 0::2], k[:, 0::2], v, mask, self.scale)
            a2 = attend(q[:, 1::2], k[:, 1::2], v, mask, self.scale)
            out = a1.float() - self.lam() * a2.float()                        # (B, H/2, T, 2*hd)
            out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.norm_eps)
            out = (out * self.subln.float()[None, :, None, :] * (1 - self.lambda_init)).type_as(v)
        return out.transpose(1, 2).reshape(B, T, -1)

    def forward(self, x, cos, sin, mask=None, kv_cache=None):
        q, k, v = self.project(x, cos, sin)
        if kv_cache is not None:                  # generation: one document, plain causal (+ window)
            k, v = kv_cache.update(k, v)
            T, Tk = q.size(2), k.size(2)
            qi = torch.arange(Tk - T, Tk, device=x.device)[:, None]
            kj = torch.arange(Tk, device=x.device)[None, :]
            mask = qi >= kj
            if self.local:
                mask = mask & (qi - kj < self.window)
        return self.o_proj(self.combine(q, k, v, mask))


class Block(nn.Module):
    def __init__(self, cfg: ArchEConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = Attention(cfg, layer_idx)
        self.ffn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, mask=None, kv_cache=None):
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin, mask, kv_cache))
        return x + self.drop(self.ffn(self.ffn_norm(x)))


class ArchEDiff(GenerationMixin, nn.Module):
    def __init__(self, cfg: ArchEConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.padded_vocab, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.padded_vocab, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._block_masks = {}
        self.doc_attn_impl = "flex"              # "dense": boolean masks + SDPA (tests compare the two)
        init_weights(self, cfg.n_layers)          # N(0, 0.02); the tied matrix is initialised once

    def lambda_values(self):
        return [b.attn.lam().item() if b.attn.diff else None for b in self.blocks]

    def causal_masks(self, T, device):
        """{local: mask} without doc_mask: causal (global) and causal + window (local)."""
        window = self.cfg.window
        if self.doc_attn_impl == "dense":
            i = torch.arange(T, device=device)
            glob = i[:, None] >= i[None, :]
            return {False: glob, True: glob & (i[:, None] - i[None, :] < window)}
        key = (T, str(device))
        if key not in self._block_masks:
            def causal(b, h, q, kv):
                return q >= kv

            def local_causal(b, h, q, kv):
                return (q >= kv) & (q - kv < window)
            self._block_masks[key] = {
                False: create_block_mask(causal, B=None, H=None, Q_LEN=T, KV_LEN=T, device=device),
                True: create_block_mask(local_causal, B=None, H=None, Q_LEN=T, KV_LEN=T, device=device)}
        return self._block_masks[key]

    def doc_masks(self, doc_ids):
        """{local: mask} with doc_mask, built once per batch (each row has its own documents)."""
        B, T = doc_ids.shape
        window = self.cfg.window
        kinds = {b.attn.local for b in self.blocks}
        if self.doc_attn_impl == "dense":
            i = torch.arange(T, device=doc_ids.device)
            glob = ((doc_ids[:, :, None] == doc_ids[:, None, :]) & (i[:, None] >= i[None, :]))[:, None]
            return {False: glob, True: glob & (i[:, None] - i[None, :] < window)}
        local_mod, global_mod = ArchDSliding.doc_mask_mods(doc_ids, window)
        return {local: create_block_mask(local_mod if local else global_mod, B=B, H=None, Q_LEN=T, KV_LEN=T,
                                         device=doc_ids.device) for local in kinds}

    def forward(self, idx, targets=None):
        T = idx.size(1)
        x = self.embed(idx)
        if self.cfg.doc_mask:
            doc_ids, pos_ids = doc_structure(idx, self.cfg.eos_id)
            cos, sin = self.rope_cos[pos_ids], self.rope_sin[pos_ids]         # (B, T, head_dim/2)
            masks = self.doc_masks(doc_ids)
        else:
            cos, sin = self.rope_cos[:T], self.rope_sin[:T]
            masks = self.causal_masks(T, idx.device)
        for block in self.blocks:
            x = block(x, cos, sin, masks[block.attn.local])
        logits = self.lm_head(self.norm(x))[..., :self.cfg.vocab_size]       # drop vocabulary padding
        return logits, (lm_loss(logits, targets) if targets is not None else None)
