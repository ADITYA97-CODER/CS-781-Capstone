"""Sentence classifier on top of a Phase-1 decoder: the Phase-2 bidirectional encoder (phase2/ner_model.Tagger),
mean-pooled over each sentence's tokens, + a relative-position-in-document embedding (+ optional domain embedding),
-> dropout -> linear (or, with mlp=True, the Laya MLP) over the 13 rhetorical roles."""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase2"))
from ner_model import Tagger, load_backbone  # noqa: E402,F401


class SentClassifier(Tagger):
    def __init__(self, backbone, n_labels, n_pos=16, domain_emb=False, bidirectional=True, dropout=0.1, mlp=False):
        super().__init__(backbone, n_labels, bidirectional=bidirectional, dropout=dropout)
        d = backbone.cfg.d_model
        if mlp:                                    # the Laya head's MLP (768 -> GELU -> 13) on the pooled vector
            self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, n_labels))
            nn.init.normal_(self.head[3].weight, std=0.02)
            nn.init.zeros_(self.head[3].bias)
        self.pos_emb = nn.Embedding(n_pos, d) if n_pos > 1 else None
        self.dom_emb = nn.Embedding(2, d) if domain_emb else None
        for e in (self.pos_emb, self.dom_emb):
            if e is not None:
                nn.init.zeros_(e.weight)

    def sentence_vectors(self, ids, valid, sid, pos, dom, n_slots):
        h = self.hidden(ids, valid)                                        # (B, T, d)
        slots = torch.arange(n_slots, device=ids.device)
        onehot = (sid[:, None, :] == slots[None, :, None]).to(h.dtype)    # (B, S, T)
        v = torch.bmm(onehot, h) / onehot.sum(-1, keepdim=True).clamp(min=1)
        if self.pos_emb is not None:
            v = v + self.pos_emb(pos)
        if self.dom_emb is not None:
            v = v + self.dom_emb(dom)[:, None]
        return v

    def forward(self, ids, valid, sid, pos, dom, n_slots, mpos=None):
        return self.head(self.drop(self.sentence_vectors(ids, valid, sid, pos, dom, n_slots)))   # (B, S, K)


class LayaClassifier(Tagger):
    """Laya-style head (as the A1 critic): a learned marker token follows every sentence and each sentence is classified
    from its own marker's final hidden state by an MLP shared across sentences. The marker input embedding is one
    learned vector (initialised from <|endoftext|>, the boundary token the model knows) + a relative-position-in-document
    embedding (+ optional domain embedding), so position conditions the whole encoder rather than only the head."""

    def __init__(self, backbone, n_labels, n_pos=16, domain_emb=False, bidirectional=True, dropout=0.1, eos_id=0):
        super().__init__(backbone, n_labels, bidirectional=bidirectional, dropout=dropout)
        d = backbone.cfg.d_model
        self.marker = nn.Parameter(backbone.embed.weight[eos_id].detach().clone())
        self.pos_emb = nn.Embedding(max(n_pos, 1), d)
        self.dom_emb = nn.Embedding(2, d) if domain_emb else None
        for e in (self.pos_emb, self.dom_emb):
            if e is not None:
                nn.init.zeros_(e.weight)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, n_labels))
        nn.init.normal_(self.head[3].weight, std=0.02)
        nn.init.zeros_(self.head[3].bias)

    def marker_vectors(self, ids, valid, pos, dom, mpos):
        bb = self.backbone
        T = ids.size(1)
        x = bb.embed(ids)
        m = self.marker + self.pos_emb(pos)                                # (B, S, d)
        if self.dom_emb is not None:
            m = m + self.dom_emb(dom)[:, None]
        ok = mpos >= 0
        b = torch.arange(ids.size(0), device=ids.device)[:, None].expand_as(mpos)
        x = x.index_put((b[ok], mpos[ok]), m[ok].to(x.dtype))
        cos, sin = bb.rope_cos[:T], bb.rope_sin[:T]
        masks = self.masks(valid)
        for block in bb.blocks:
            x = block(x, cos, sin, masks[block.attn.local])
        h = bb.norm(x)
        return h.gather(1, mpos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1)))

    def forward(self, ids, valid, sid, pos, dom, n_slots, mpos=None):
        return self.head(self.drop(self.marker_vectors(ids, valid, pos, dom, mpos)))      # (B, S, K)
