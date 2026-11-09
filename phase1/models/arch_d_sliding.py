"""Arch D: efficient decoder with sliding-window attention and grouped-query attention.

  * Local layers: token t attends only to tokens t-window+1 .. t (window=512), so attention
    cost grows with seq_len * window instead of seq_len^2. Implemented with FlexAttention,
    which skips the blocks of the attention matrix outside the window.
  * Global layers: every `global_every`-th layer (layers 4, 8, 12 of 12) uses full causal
    attention, so information from far back can still reach every position.
  * Grouped-query attention: n_heads query heads share n_kv_heads key/value heads
    (10 queries : 2 key/value heads), shrinking K/V projections and the KV cache 5x.

  * doc_mask (optional, default off): packed rows hold several documents separated by
    <|endoftext|>. With doc_mask, a token attends only to earlier tokens of its own document
    (in local and global layers) and RoPE positions restart at 0 in every document.

Default size (configs/archD.json): d=640, 12 layers, 10 query / 2 KV heads, SwiGLU 1728 -> 72M params.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    HAS_FLEX = True
except ImportError:          # doc_mask then uses dense boolean masks with scaled_dot_product_attention
    HAS_FLEX = False

from .layers import GenerationMixin, SwiGLU, apply_rope, doc_structure, init_weights, lm_loss, rope_tables


@dataclass
class ArchDConfig:
    arch: str = "D"
    vocab_size: int = 32000
    d_model: int = 640
    n_layers: int = 12
    n_heads: int = 10
    n_kv_heads: int = 2
    ffn_hidden: int = 1728
    max_seq_len: int = 1536
    window: int = 512
    global_every: int = 4
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    doc_mask: bool = False      # intra-document attention + per-document RoPE positions
    eos_id: int = 0             # <|endoftext|> in legal_bpe_32k

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0 and self.n_heads % self.n_kv_heads == 0

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    def is_local(self, layer_idx):
        return (layer_idx + 1) % self.global_every != 0


class GQAttention(nn.Module):
    """Grouped-query attention; `local=True` restricts it to a causal sliding window."""

    def __init__(self, cfg: ArchDConfig, local: bool):
        super().__init__()
        self.local, self.window = local, cfg.window
        self.n_heads, self.n_kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=False)   # fewer K heads
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=False)   # fewer V heads
        self.o_proj = nn.Linear(cfg.n_heads * self.hd, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, block_mask=None, kv_cache=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if kv_cache is None:                                    # training
            if block_mask is None:                              # global layer without doc_mask
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            elif isinstance(block_mask, torch.Tensor):          # doc_mask dense fallback, (B, 1, T, T)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=block_mask, enable_gqa=True)
            else:
                out = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)
        else:                                                   # generation
            k, v = kv_cache.update(k, v)
            if T > 1:                                           # prompt prefill
                mask = None
                if self.local:                                  # same local-causal pattern as training
                    i = torch.arange(T, device=x.device)
                    mask = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < self.window)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None,
                                                     enable_gqa=True)
            else:                                               # one new token
                if self.local:
                    k, v = k[:, :, -self.window:], v[:, :, -self.window:]
                out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


class Block(nn.Module):
    def __init__(self, cfg: ArchDConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = GQAttention(cfg, local=cfg.is_local(layer_idx))
        self.ffn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, block_mask=None, kv_cache=None):
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin, block_mask, kv_cache))
        return x + self.drop(self.ffn(self.ffn_norm(x)))


class ArchDSliding(GenerationMixin, nn.Module):
    def __init__(self, cfg: ArchDConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._block_masks = {}
        self.doc_attn_impl = "flex" if HAS_FLEX else "dense"   # tests switch this to compare the two
        init_weights(self, cfg.n_layers)

    def sliding_block_mask(self, seq_len, device):
        """FlexAttention block mask for 'causal and within the window' (cached per length)."""
        key = (seq_len, str(device))
        if key not in self._block_masks:
            window = self.cfg.window

            def local_causal(b, h, q_idx, kv_idx):
                return (q_idx >= kv_idx) & (q_idx - kv_idx < window)

            self._block_masks[key] = create_block_mask(local_causal, B=None, H=None,
                                                       Q_LEN=seq_len, KV_LEN=seq_len, device=device)
        return self._block_masks[key]

    @staticmethod
    def doc_mask_mods(doc_ids, window):
        """mask_mods for doc_mask: same document and causal (global layers), plus within the
        window (local layers). Each batch row b uses its own doc_ids[b]."""
        def doc_causal(b, h, q_idx, kv_idx):
            return (doc_ids[b, q_idx] == doc_ids[b, kv_idx]) & (q_idx >= kv_idx)

        def doc_local(b, h, q_idx, kv_idx):
            return doc_causal(b, h, q_idx, kv_idx) & (q_idx - kv_idx < window)
        return doc_local, doc_causal

    def doc_block_masks(self, doc_ids):
        """(local, global) masks for one batch, built once and shared by all layers of each kind:
        FlexAttention BlockMasks, or dense (B, 1, T, T) boolean masks for the SDPA fallback."""
        B, T = doc_ids.shape
        window = self.cfg.window
        if self.doc_attn_impl == "dense":
            i = torch.arange(T, device=doc_ids.device)
            glob = ((doc_ids[:, :, None] == doc_ids[:, None, :]) & (i[:, None] >= i[None, :]))[:, None]
            return glob & (i[:, None] - i[None, :] < window), glob
        local_mod, global_mod = self.doc_mask_mods(doc_ids, window)
        return tuple(create_block_mask(mod, B=B, H=None, Q_LEN=T, KV_LEN=T, device=doc_ids.device)
                     for mod in (local_mod, global_mod))

    def forward(self, idx, targets=None):
        T = idx.size(1)
        x = self.embed(idx)
        if self.cfg.doc_mask:
            doc_ids, pos_ids = doc_structure(idx, self.cfg.eos_id)
            cos, sin = self.rope_cos[pos_ids], self.rope_sin[pos_ids]         # (B, T, head_dim/2)
            local_mask, global_mask = self.doc_block_masks(doc_ids)
        else:
            cos, sin = self.rope_cos[:T], self.rope_sin[:T]
            local_mask, global_mask = self.sliding_block_mask(T, idx.device), None
        for block in self.blocks:
            x = block(x, cos, sin, local_mask if block.attn.local else global_mask)
        logits = self.lm_head(self.norm(x))
        return logits, (lm_loss(logits, targets) if targets is not None else None)
