"""Document classifier for Court Judgment Prediction on top of a Phase-1 decoder (Arch E / F).

Encoder: the Phase-1 blocks, unchanged, with a bidirectional mask. Every document is cut into one or more chunks
(cjpe_data.doc_chunks) and the chunks are packed into fixed-size rows (B, T): a chunk attends to every token of its own
chunk (local layers: within |q - k| < window), RoPE positions restart at 0 in every chunk. Masks are FlexAttention
BlockMasks built once per batch from the per-token segment ids, so packing several short chunks into one row costs
no padding and the shapes stay static for torch.compile.

Head: mean over each chunk's tokens -> chunk vector (+ relative-position-in-document bucket embedding) -> document
vector by pooling the document's chunk vectors ('mean', or 'attn': softmax of a learned score, Yang et al. 2016-style
attention pooling) -> dropout -> linear (2 classes). With one chunk per document the document pooling is the identity.
"""

import os
import sys

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phase1"))
from models import build_model  # noqa: E402
from models.layers import rope_tables  # noqa: E402


def load_backbone(ckpt_path, max_len=4096):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, cfg = build_model(state["config"]["model"])
    model.load_state_dict(state["model"])
    if max_len > model.rope_cos.size(0):          # chunks longer than pretraining (1536)
        cos, sin = rope_tables(cfg.head_dim, max_len, cfg.rope_theta)
        model.rope_cos, model.rope_sin = cos, sin
    return model, cfg


def segment_masks(seg, window):
    """seg: (B, T) segment id per token (padding has its own id) -> {local: BlockMask}, bidirectional within segments."""
    B, T = seg.shape

    def same(b, h, q, kv):
        return seg[b, q] == seg[b, kv]

    def local(b, h, q, kv):
        return (seg[b, q] == seg[b, kv]) & ((q - kv).abs() < window)
    return {False: create_block_mask(same, B=B, H=None, Q_LEN=T, KV_LEN=T, device=seg.device),
            True: create_block_mask(local, B=B, H=None, Q_LEN=T, KV_LEN=T, device=seg.device)}


class Encoder(nn.Module):
    """The Phase-1 blocks with packed bidirectional segments. Compiled separately (static (B, T))."""

    def __init__(self, backbone, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        for block in backbone.blocks:
            block.drop.p = dropout
        backbone.lm_head = None                     # the tied LM head is not used

    def forward(self, ids, pos, masks):
        bb = self.backbone
        x = bb.embed(ids)
        cos, sin = bb.rope_cos[pos], bb.rope_sin[pos]                      # (B, T, hd/2)
        for block in bb.blocks:
            x = block(x, cos, sin, masks[block.attn.local])
        return bb.norm(x)


class DocClassifier(nn.Module):
    def __init__(self, backbone, n_classes=2, dropout=0.1, doc_pool="mean", n_pos=8, mlp=False):
        super().__init__()
        self.encoder = Encoder(backbone, dropout)
        self.window = backbone.cfg.window
        d = backbone.cfg.d_model
        self.doc_pool = doc_pool
        self.drop = nn.Dropout(dropout)
        self.pos_emb = nn.Embedding(n_pos, d) if n_pos > 1 else None
        if self.pos_emb is not None:
            nn.init.zeros_(self.pos_emb.weight)
        if doc_pool == "attn":
            self.att = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
            nn.init.zeros_(self.att[2].weight)        # starts as the mean
            nn.init.zeros_(self.att[2].bias)
        if mlp:
            self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, n_classes))
            last = self.head[3]
        else:
            self.head = last = nn.Linear(d, n_classes)
        nn.init.normal_(last.weight, std=0.02)
        nn.init.zeros_(last.bias)
        self.compiled = None                        # set_compiled(): torch.compile(self.encoder), kept out of the modules

    def set_compiled(self, fn):
        object.__setattr__(self, "compiled", fn)

    def encode(self, ids, pos, masks):
        return (self.encoder if self.compiled is None else self.compiled)(ids, pos, masks)

    @property
    def backbone(self):
        return self.encoder.backbone

    def doc_vectors(self, b):
        """b: a packed batch (cjpe_data.pack) on the device -> (n_docs, d)."""
        masks = segment_masks(b["seg"], self.window)
        h = self.encode(b["ids"], b["pos"], masks).float()                # (B, T, d)
        d = h.size(-1)
        tok = b["tok_chunk"].view(-1)                                       # chunk index per token, -1 = padding
        keep = tok >= 0
        n_chunks = b["chunk_doc"].numel()
        csum = h.new_zeros(n_chunks, d).index_add_(0, tok[keep], h.view(-1, d)[keep])
        cnt = h.new_zeros(n_chunks).index_add_(0, tok[keep], h.new_ones(int(keep.sum())))
        c = csum / cnt.clamp(min=1)[:, None]                                # (n_chunks, d)
        if self.pos_emb is not None:
            c = c + self.pos_emb(b["chunk_pos"])
        slots = b["doc_slots"]                                              # (n_docs, K) chunk indices, -1 = none
        ok = slots >= 0
        cv = c[slots.clamp(min=0)]                                          # (n_docs, K, d)
        if self.doc_pool == "attn":
            w = self.att(cv).squeeze(-1).masked_fill(~ok, float("-inf")).softmax(-1)
        else:                                                               # mean (max over chunks not needed)
            w = ok.float() / ok.sum(-1, keepdim=True)
        return (w[..., None] * cv).sum(1)

    def forward(self, b):
        return self.head(self.drop(self.doc_vectors(b)))                    # (n_docs, n_classes) fp32
