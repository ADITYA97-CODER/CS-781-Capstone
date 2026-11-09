"""Label-wise attention head (LWAN, Mullenbach et al. 2018 / Chalkidis et al. 2019 style) on the phase4 DocClassifier.

Every statute l has a query u_l and an output vector w_l. Over all tokens t of a document (all its chunks):
    a_lt = softmax_t(u_l . h_t),   logit_l = sum_t a_lt (w_l . h_t) + b_l   (+ the pooled-document head's logit_l)
computed as two (n_tokens, L) matrices and segment sums, so no (n_docs, L, d) tensor is formed. u starts at 0
(uniform attention = token mean) and w at 0, so the model starts as the plain DocClassifier.
"""

import torch
import torch.nn as nn

from cjpe_model import DocClassifier, segment_masks


class LWANClassifier(DocClassifier):
    def __init__(self, backbone, n_classes, dropout=0.1, doc_pool="attn", n_pos=8, mlp=False):
        super().__init__(backbone, n_classes, dropout, doc_pool, n_pos, mlp)
        d = backbone.cfg.d_model
        self.U = nn.Parameter(torch.zeros(n_classes, d))
        self.W = nn.Parameter(torch.zeros(n_classes, d))
        self.n_classes = n_classes

    def forward(self, b):
        masks = segment_masks(b["seg"], self.window)
        h = self.encode(b["ids"], b["pos"], masks).float()                # (B, T, d)
        d = h.size(-1)
        tok = b["tok_chunk"].view(-1)
        keep = tok >= 0
        ht = h.view(-1, d)[keep]                                           # (N, d)
        chunk = tok[keep]
        n_docs = b["doc_slots"].size(0)
        n_chunks = b["chunk_doc"].numel()
        # pooled-document path (same as DocClassifier.doc_vectors)
        csum = h.new_zeros(n_chunks, d).index_add_(0, chunk, ht)
        cnt = h.new_zeros(n_chunks).index_add_(0, chunk, h.new_ones(chunk.numel()))
        c = csum / cnt.clamp(min=1)[:, None]
        if self.pos_emb is not None:
            c = c + self.pos_emb(b["chunk_pos"])
        slots = b["doc_slots"]
        ok = slots >= 0
        cv = c[slots.clamp(min=0)]
        if self.doc_pool == "attn":
            w = self.att(cv).squeeze(-1).masked_fill(~ok, float("-inf")).softmax(-1)
        else:
            w = ok.float() / ok.sum(-1, keepdim=True)
        doc_logits = self.head(self.drop((w[..., None] * cv).sum(1)))      # (n_docs, L)
        # label-wise attention over the tokens of each document
        doc = b["chunk_doc"][chunk]                                        # (N,) document of every token
        return doc_logits + self.label_attention(self.drop(ht), doc, n_docs)

    @torch.autocast("cuda", enabled=False)
    def label_attention(self, ht, doc, n_docs):
        """fp32 (outside autocast): segment softmax + sums over the tokens of each document -> (n_docs, L)."""
        ht = ht.float()
        s = ht @ self.U.float().t()                                        # (N, L)
        idx = doc[:, None].expand_as(s)
        m = s.new_full((n_docs, self.n_classes), -1e30).scatter_reduce(0, idx, s, "amax", include_self=True)
        e = torch.exp(s - m[doc])
        den = s.new_zeros(n_docs, self.n_classes).index_add_(0, doc, e)
        g = ht @ self.W.float().t()                                        # (N, L)
        return s.new_zeros(n_docs, self.n_classes).index_add_(0, doc, e * g) / den.clamp(min=1e-30)
