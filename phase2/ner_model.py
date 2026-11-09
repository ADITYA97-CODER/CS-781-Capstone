"""Token tagger on top of a Phase-1 decoder (Arch E / F).

The Phase-1 blocks are reused unchanged; only the attention mask differs from pretraining:
  * bidirectional=True: every token attends to every real (non-pad) token of its window; local
    layers attend within |q - k| < window. bidirectional=False keeps the pretraining causal mask.
  * Masks are dense booleans for SDPA (one window per row, so no document structure is needed),
    which avoids the FlexAttention + torch.compile recompilation trap.
Positions are 0..T-1 in every window (RoPE restarts per window, as with doc_mask).
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1"))
from models import build_model  # noqa: E402


def load_backbone(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, cfg = build_model(state["config"]["model"])
    model.load_state_dict(state["model"])
    return model, cfg


class CRF(nn.Module):
    """Linear-chain CRF over the BIO tags (Lafferty et al. 2001; BiLSTM/BERT-CRF taggers). Learned transition,
    start and end scores; transitions the BIO scheme forbids (I-X after anything but B-X / I-X) get -1e4. The
    start constraint is not applied in training because a content window can begin inside an entity (I-X).
    Emissions are the tagger's logits; at test time `ner_data.viterbi` adds the exported scores."""

    def __init__(self, n_tags, allowed):
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(n_tags, n_tags))
        self.start = nn.Parameter(torch.zeros(n_tags))
        self.end = nn.Parameter(torch.zeros(n_tags))
        self.register_buffer("forbid", torch.where(torch.as_tensor(allowed), 0.0, -1e4), persistent=False)

    def export(self):
        return tuple(x.detach().float().cpu().numpy() for x in (self.trans, self.start, self.end))

    def nll(self, emissions, labels):
        """emissions (B, T, K) float32, labels (B, T) with -100 outside one contiguous labelled segment per row.
        -> negative log-likelihood summed over rows / number of labelled tokens."""
        lab_ok = labels >= 0
        L = lab_ok.sum(1)                                           # (B,)
        keep = L > 0
        emissions, labels, lab_ok, L = emissions[keep], labels[keep], lab_ok[keep], L[keep]
        B, T, K = emissions.shape
        s = lab_ok.float().argmax(1)                                # first labelled position
        j = torch.arange(int(L.max()), device=emissions.device)
        idx = (s[:, None] + j[None]).clamp(max=T - 1)
        m = j[None] < L[:, None]                                    # (B, Lmax)
        em = emissions.gather(1, idx[:, :, None].expand(-1, -1, K))
        tags = labels.gather(1, idx).clamp(min=0)
        trans = self.trans + self.forbid
        # gold path score
        gold = self.start[tags[:, 0]] + em[:, 0].gather(1, tags[:, :1]).squeeze(1)
        e_t = em.gather(2, tags[:, :, None]).squeeze(2)            # (B, Lmax)
        tr = trans[tags[:, :-1], tags[:, 1:]]                       # (B, Lmax - 1)
        gold = gold + ((e_t[:, 1:] + tr) * m[:, 1:]).sum(1)
        last = tags.gather(1, (L - 1)[:, None]).squeeze(1)
        gold = gold + self.end[last]
        # partition function
        alpha = self.start[None] + em[:, 0]
        for t in range(1, em.size(1)):
            nxt = torch.logsumexp(alpha[:, :, None] + trans[None], 1) + em[:, t]
            alpha = torch.where(m[:, t, None], nxt, alpha)
        logz = torch.logsumexp(alpha + self.end[None], 1)
        return (logz - gold).sum() / L.sum()


class Tagger(nn.Module):
    def __init__(self, backbone, n_tags, bidirectional=True, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.bidirectional = bidirectional
        for block in backbone.blocks:
            block.drop.p = dropout
        backbone.lm_head = None                    # the tied LM head is not used
        d = backbone.cfg.d_model
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d, n_tags)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def masks(self, valid):
        """valid: (B, T) real tokens -> {local: (B, 1, T, T) bool}."""
        B, T = valid.shape
        i = torch.arange(T, device=valid.device)
        keys = valid[:, None, None, :]
        if self.bidirectional:
            glob = keys.expand(B, 1, T, T)
            dist = (i[:, None] - i[None, :]).abs()
        else:
            glob = keys & (i[:, None] >= i[None, :])
            dist = i[:, None] - i[None, :]
        return {False: glob, True: glob & (dist < self.backbone.cfg.window)}

    def hidden(self, ids, valid):
        bb = self.backbone
        T = ids.size(1)
        x = bb.embed(ids)
        cos, sin = bb.rope_cos[:T], bb.rope_sin[:T]
        masks = self.masks(valid)
        for block in bb.blocks:
            x = block(x, cos, sin, masks[block.attn.local])
        return bb.norm(x)

    def forward(self, ids, valid):
        return self.head(self.drop(self.hidden(ids, valid)))
