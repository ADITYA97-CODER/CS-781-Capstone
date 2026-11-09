"""Span-classification head for L-NER (alternative to the BIO token head).

Every candidate span (i, j) of content tokens with j - i < max_len is classified as O or one of the 12 labels:
    z(i, j) = GELU( P_s h_i + P_e h_j + W_width[j - i] )        (hidden size H)
    logits(i, j) = U z(i, j)                                    (13 classes: 0 = O, 1..12 = LABELS)
Start and end are projected separately and added, so all B x T x max_len candidates are scored without forming
(T x T x 2d) concatenations. The span's own tokens are seen through h_i / h_j, which already attend to the whole
window (bidirectional backbone).

Training: cross-entropy over all valid candidates (gold spans -> their label, every other span -> O).
Decoding: per window, keep candidates whose best entity probability > `keep`; across overlapping windows keep each
span's scores from the window where it sits furthest from an edge; average across seeds/models; then greedily take
non-overlapping spans in order of entity probability while it exceeds `threshold`. Token spans -> characters with
the same snap / trim / propagation as the BIO decoder.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ner_data as D

SPAN_LABELS = ["O"] + D.LABELS
LABEL2ID = {l: i for i, l in enumerate(SPAN_LABELS)}


class SpanHead(nn.Module):
    def __init__(self, d, max_len=48, hidden=256, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.p_start = nn.Linear(d, hidden)
        self.p_end = nn.Linear(d, hidden, bias=False)
        self.width = nn.Parameter(torch.zeros(max_len, hidden))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, len(SPAN_LABELS))
        for lin in (self.p_start, self.p_end, self.out):
            nn.init.normal_(lin.weight, std=0.02)
        nn.init.normal_(self.width, std=0.02)
        nn.init.zeros_(self.p_start.bias)
        nn.init.zeros_(self.out.bias)

    def forward(self, h, valid):
        """h: (B, Tc, d) content hidden states; valid: (B, Tc) -> logits (B, Tc, L, 13), span_ok (B, Tc, L)."""
        B, T, _ = h.shape
        L = self.max_len
        zs = self.p_start(h)                                            # (B, T, H)
        ze = F.pad(self.p_end(h), (0, 0, 0, L))                         # (B, T + L, H)
        ze = ze.unfold(1, L, 1)[:, :T].permute(0, 1, 3, 2)               # (B, T, L, H): end = i + w
        z = F.gelu(zs[:, :, None] + ze + self.width[None, None])
        logits = self.out(self.drop(z))
        vend = F.pad(valid, (0, L)).unfold(1, L, 1)[:, :T]               # (B, T, L): end token valid
        return logits, valid[:, :, None] & vend


def span_labels(rows_spans, T, L, span_ok):
    """rows_spans: per row, [(i, j, label)] in content coordinates -> (B, T, L) targets (-100 where invalid)."""
    lab = torch.zeros(span_ok.shape, dtype=torch.long)
    for b, spans in enumerate(rows_spans):
        for i, j, label in spans:
            if j - i < L:
                lab[b, i, j - i] = LABEL2ID[label]
    lab[~span_ok.cpu()] = -100
    return lab


@torch.no_grad()
def predict_candidates(model, win, device, batch_size, keep=0.01):
    """-> per window row: (i, j, probs (n, 13)) arrays in content coordinates, for spans with P(entity) > keep."""
    model.eval()
    c0 = win.header_len + 1
    out = []
    for r in range(0, len(win), batch_size):
        ids = win.ids[r:r + batch_size].to(device)
        valid = win.valid[r:r + batch_size].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.hidden(ids, valid)[:, c0:]
            logits, ok = model.head(h, valid[:, c0:])
        probs = logits.float().softmax(-1)
        ent = (1 - probs[..., 0]) * ok
        for b in range(ids.size(0)):
            ii, ww = torch.nonzero(ent[b] > keep, as_tuple=True)
            out.append((ii.cpu().numpy(), (ii + ww).cpu().numpy(), probs[b, ii, ww].cpu().numpy()))
    model.train()
    return out


def merge_candidates(n_tokens_per_doc, meta, row_cands):
    """-> per document {(i, j): probs} in document token coordinates (best-centred window wins)."""
    docs = [dict() for _ in n_tokens_per_doc]
    best = [dict() for _ in n_tokens_per_doc]
    for (di, lo, hi, _), (ii, jj, pp) in zip(meta, row_cands):
        n = n_tokens_per_doc[di]
        for i, j, p in zip(ii, jj, pp):
            a, b = lo + int(i), lo + int(j)
            left = 10**9 if lo == 0 else a - lo
            right = 10**9 if hi == n else hi - 1 - b
            dist = min(left, right)
            if dist > best[di].get((a, b), -1):
                best[di][(a, b)] = dist
                docs[di][(a, b)] = p
    return docs


def average_candidates(cand_list):
    """Average {(i, j): probs} dicts of several models; a span missing from a model counts as O."""
    keys = set().union(*cand_list)
    o = np.zeros(len(SPAN_LABELS), dtype=np.float32)
    o[0] = 1.0
    return {k: np.mean([c.get(k, o) for c in cand_list], axis=0) for k in keys}


def decode_doc_spans(text, offs, cands, threshold=0.5, propagate=True):
    """{(i, j): probs} -> [[start, end, label], ...]: greedy non-overlapping spans by entity probability."""
    scored = []
    for (i, j), p in cands.items():
        k = int(np.argmax(p[1:])) + 1
        if p[k] > threshold:
            scored.append((float(p[k]), i, j, SPAN_LABELS[k]))
    scored.sort(reverse=True)
    used = np.zeros(len(offs) + 1, dtype=bool)
    out, seen = [], set()
    for _, i, j, label in scored:
        if used[i:j + 1].any():
            continue
        used[i:j + 1] = True
        s, e = D.snap_words(text, int(offs[i, 0]), int(offs[j, 1]))
        s, e = D.trim_end(text, s, e, label)
        if e - s >= 2 and (s, e, label) not in seen:
            seen.add((s, e, label))
            out.append([s, e, label])
    out.sort()
    return D.propagate_doc(text, out) if propagate else out


def save_candidates(path, doc_ids, docs_cands):
    arrs = {}
    for k, c in zip(doc_ids, docs_cands):
        keys = np.array(list(c.keys()), dtype=np.int64).reshape(-1, 2)
        arrs[f"{k}__ij"] = keys
        arrs[f"{k}__p"] = np.array([c[tuple(x)] for x in keys], dtype=np.float32).reshape(-1, len(SPAN_LABELS))
    np.savez_compressed(path, **arrs)


def load_candidates(path):
    z = np.load(path)
    out = {}
    for name in z.files:
        if name.endswith("__ij"):
            k = name[:-4]
            out[k] = {(int(a), int(b)): p for (a, b), p in zip(z[name], z[k + "__p"])}
    return out
