"""Phase 3 (Rhetorical Role) data: documents -> sentence-packed token windows, and window scores -> sentence labels.

  * Every sentence is encoded on its own with the Phase-1 tokenizer (legal_bpe_32k) as " " + sentence, so that a
    run of sentences reads as running text, and truncated to `max_sent_tok` tokens (99th percentile is ~150-190).
  * ctx='window': a row is a run of consecutive whole sentences of one document, packed up to `max_len` tokens.
    Training rows overlap (each new row starts about `stride` tokens after the previous one); at prediction time a
    sentence keeps the scores of the row where its centre is furthest from a row edge.
  * ctx='none': a row is a single sentence (the sentence-only baseline).
  * Each sentence also gets its relative position in the document, as one of `n_pos` buckets.
  * marker=True (Laya-style head): one marker slot follows every sentence (id 0 as a placeholder; the model swaps
    in a learned marker embedding there) and `mpos` gives each sentence's marker position.
"""

import json
import os

import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "Phase_3_RR")
TOKENIZER = os.path.join(ROOT, "artifacts", "tokenizer", "legal_bpe_32k", "tokenizer.json")
_vocab = json.load(open(os.path.join(DATA, "Train", "label_vocab.json")))
LABELS = sorted(_vocab, key=_vocab.get)
LAB2ID = {l: i for i, l in enumerate(LABELS)}
DOMAINS = ("CL", "IT")


def tokenizer():
    return Tokenizer.from_file(TOKENIZER)


def cv_folds(k):
    """All labelled docs (train + dev, both domains) -> k folds of ids. Per domain, docs are taken largest first and
    each goes to the fold with the fewest sentences of that domain so far (ties: lowest fold), so folds are balanced
    in sentences per domain. Docs containing Dissent (2 CL docs) are placed first, so they go to different folds and
    Dissent is both trained and tested; PrecedentOverruled (3 sentences, 1 doc) is never in both. Deterministic."""
    folds = [[] for _ in range(k)]
    for dom in DOMAINS:
        docs = load_split(f"{dom}_train") + load_split(f"{dom}_dev")
        size = [0] * k
        for d in sorted(docs, key=lambda d: ("Dissent" not in d["labels"], -len(d["text"]), d["id"])):
            f = min(range(k), key=lambda i: (size[i], i))
            folds[f].append(d["id"])
            size[f] += len(d["text"])
    return folds


def load_split(name):
    """'CL_train' | 'IT_dev' | 'CL_test' ... -> list of dicts {id, domain, text, labels (None for test)}.
    'cv<k>_f<i>' -> the labelled docs of fold i of k (cv_folds)."""
    if name.startswith("cv"):
        k, i = name[2:].split("_f")
        keep = set(cv_folds(int(k))[int(i)])
        return [d for dom in DOMAINS for part in ("train", "dev") for d in load_split(f"{dom}_{part}") if d["id"] in keep]
    dom, part = name.split("_")
    sub = "Test" if part == "test" else "Train"
    raw = json.load(open(os.path.join(DATA, sub, f"{name}.json")))
    if isinstance(raw, dict):                                   # test files: {id: [sentences]}
        raw = [{"id": k, "text": v} for k, v in raw.items()]
    docs = []
    for x in raw:
        labels = x.get("labels")
        if labels is not None:
            assert len(labels) == len(x["text"]), x["id"]
        docs.append({"id": x["id"], "domain": dom, "text": x["text"], "labels": labels})
    return docs


def encode_docs(tok, docs, max_sent_tok):
    for d in docs:
        enc = tok.encode_batch([" " + s.strip() for s in d["text"]], add_special_tokens=False)
        d["ids"] = [np.array(e.ids[:max_sent_tok] or [0], dtype=np.int64) for e in enc]
        d["y"] = None if d["labels"] is None else np.array([LAB2ID[l] for l in d["labels"]], dtype=np.int64)
    return docs


def pos_bucket(i, n, n_pos):
    return min(n_pos - 1, int(n_pos * i / max(1, n)))


def window_spans(lens, max_len, stride):
    """Sentence token lengths -> list of (first, last + 1) runs of whole sentences of <= max_len tokens that
    cover every sentence; consecutive runs start about `stride` tokens apart (stride >= max_len: no overlap)."""
    n = len(lens)
    cum = np.concatenate([[0], np.cumsum(lens)])
    spans, s = [], 0
    while True:
        e = s + 1
        while e < n and cum[e + 1] - cum[s] <= max_len:
            e += 1
        spans.append((s, e))
        if e >= n:
            return spans
        nxt = int(np.searchsorted(cum, cum[s] + stride, side="left"))
        s = min(max(s + 1, nxt), e)                             # never skip a sentence


class SentWindows:
    """Fixed-size tensors for a list of encoded docs.
    ids/valid (R, T); sid (R, T) local sentence slot or -1; y (R, S) label or -100; pos (R, S) bucket; dom (R,).
    mpos (R, S) marker position or -1 (marker=True only).
    meta[r] = [(doc_index, sent_index, centre_token, row_len), ...] per slot."""

    def __init__(self, docs, ctx, max_len, stride, n_pos, marker=False):
        rows = []
        mk = int(marker)
        for di, d in enumerate(docs):
            lens = [len(x) + mk for x in d["ids"]]
            spans = [(i, i + 1) for i in range(len(lens))] if ctx == "none" else window_spans(lens, max_len, stride)
            for s, e in spans:
                rows.append((di, s, e))
        T = max(sum(len(docs[di]["ids"][i]) + mk for i in range(s, e)) for di, s, e in rows)
        S = max(e - s for _, s, e in rows)
        R = len(rows)
        self.ids = torch.zeros(R, T, dtype=torch.long)
        self.valid = torch.zeros(R, T, dtype=torch.bool)
        self.sid = torch.full((R, T), -1, dtype=torch.long)
        self.y = torch.full((R, S), -100, dtype=torch.long)
        self.pos = torch.zeros(R, S, dtype=torch.long)
        self.dom = torch.zeros(R, dtype=torch.long)
        self.mpos = torch.full((R, S), -1, dtype=torch.long)
        self.meta = []
        for r, (di, s, e) in enumerate(rows):
            d, t, m = docs[di], 0, []
            n = len(d["ids"])
            for j, i in enumerate(range(s, e)):
                x = d["ids"][i]
                self.ids[r, t:t + len(x)] = torch.from_numpy(x)
                self.sid[r, t:t + len(x)] = j
                if d["y"] is not None:
                    self.y[r, j] = int(d["y"][i])
                self.pos[r, j] = pos_bucket(i, n, n_pos)
                m.append((di, i, t + len(x) / 2))
                t += len(x)
                if marker:
                    self.mpos[r, j] = t
                    t += 1
            self.valid[r, :t] = True
            self.dom[r] = DOMAINS.index(d["domain"])
            self.meta.append([(di, i, c, t) for di, i, c in m])
        self.seq_len, self.n_slots = T, S

    def __len__(self):
        return len(self.meta)


def merge_scores(docs, win, row_scores):
    """row_scores[r]: (n_slots, K) -> per doc (n_sents, K), each sentence from its most central row."""
    K = row_scores[0].shape[-1]
    out = [np.zeros((len(d["ids"]), K), dtype=np.float32) for d in docs]
    best = [np.full(len(d["ids"]), -1.0) for d in docs]
    for r, m in enumerate(win.meta):
        for j, (di, i, c, t) in enumerate(m):
            score = min(c, t - c)
            if score > best[di][i]:
                best[di][i] = score
                out[di][i] = row_scores[r][j]
    assert all((b >= 0).all() for b in best)
    return out


def macro_scores(y_true, y_pred):
    """Official metric: macro P/R/F1 over all 13 labels, zero_division=0 (a label absent from gold and never
    predicted scores 0). Also returns per-label (p, r, f1, support)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    per = {}
    for k, l in enumerate(LABELS):
        tp = int(((y_pred == k) & (y_true == k)).sum())
        fp = int(((y_pred == k) & (y_true != k)).sum())
        fn = int(((y_pred != k) & (y_true == k)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per[l] = (p, r, f, tp + fn)
    m = lambda i: float(np.mean([v[i] for v in per.values()]))  # noqa: E731
    return m(2), m(0), m(1), per
