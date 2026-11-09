"""CJPE data: splits, token cache, document views (which tokens of a judgment the model reads) and row packing.

Splits (EXPERIMENTS.md P4.1):
  train      31,677 unique cases of Train/train.json (ids that appear twice have identical text and label)
  dev_clean  the 994 cases of Train/dev.json that are NOT in the test set (dev.json contains every test case with its
             label; those rows are never loaded here)
  test       Test/test.json without its label field
"""

import json
import os

import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "Phase_4_CJPE")
CACHE = os.path.join(ROOT, "runs_p4", "cache")
TOKENIZER = os.path.join(ROOT, "artifacts", "tokenizer", "legal_bpe_32k", "tokenizer.json")
PAD_SEG = 1 << 20


def _dedup(rows):
    out, seen = [], set()
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def load_split(name):
    """-> list of {"id", "text", "label"} (label None for test)."""
    if name == "test":
        rows = json.load(open(os.path.join(DATA, "Test", "test.json")))
        return [{"id": r["id"], "text": r["text"], "label": None} for r in rows]
    if name == "train":
        rows = _dedup(json.load(open(os.path.join(DATA, "Train", "train.json"))))
    elif name == "dev_clean":
        test_ids = {r["id"] for r in json.load(open(os.path.join(DATA, "Test", "test.json")))}
        rows = _dedup([r for r in json.load(open(os.path.join(DATA, "Train", "dev.json"))) if r["id"] not in test_ids])
    else:
        raise ValueError(name)
    return [{"id": r["id"], "text": r["text"], "label": int(r["label"])} for r in rows]


def load_tokens(name):
    """-> (ids list, labels np.array (-1 for test), token arrays list of np.int32). Cached in runs_p4/cache."""
    path = os.path.join(CACHE, f"{name}.npz")
    if not os.path.exists(path):
        rows = load_split(name)
        tok = Tokenizer.from_file(TOKENIZER)
        enc = tok.encode_batch([r["text"] for r in rows])
        lens = np.array([len(e.ids) for e in enc], dtype=np.int64)
        flat = np.concatenate([np.asarray(e.ids, dtype=np.uint16) for e in enc])
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, ids=np.array([r["id"] for r in rows]), labels=np.array(
            [-1 if r["label"] is None else r["label"] for r in rows]), lens=lens, flat=flat)
    z = np.load(path)
    offs = np.concatenate([[0], np.cumsum(z["lens"])])
    flat = z["flat"].astype(np.int32)
    toks = [flat[offs[i]:offs[i + 1]] for i in range(len(z["lens"]))]
    return [str(i) for i in z["ids"]], z["labels"].astype(np.int64), toks


def doc_chunks(t, view, max_len, max_chunks=1, head_len=256):
    """Token array of one judgment -> list of (chunk tokens, start offset in the doc).
      last      the last max_len tokens (the end of the judgment, where the reasoning before the decision is)
      first     the first max_len tokens (header, parties, facts)
      headtail  first head_len + last (max_len - head_len) tokens as one chunk
      hier      the last max_chunks * max_len tokens cut into chunks of max_len (end-aligned), pooled by the head
      spread    max_chunks chunks of max_len spread evenly over the whole document (start, ..., end)
    """
    n = len(t)
    if view == "last":
        return [(t[-max_len:], max(0, n - max_len))]
    if view == "first":
        return [(t[:max_len], 0)]
    if view == "headtail":
        if n <= max_len:
            return [(t, 0)]
        return [(np.concatenate([t[:head_len], t[-(max_len - head_len):]]), 0)]
    if view == "hier":
        start = max(0, n - max_chunks * max_len)
        out, end = [], n
        while end > start:
            s = max(start, end - max_len)
            out.append((t[s:end], s))
            end = s
        return out[::-1]
    if view == "spread":
        if n <= max_chunks * max_len:
            return doc_chunks(t, "hier", max_len, max_chunks)
        starts = np.linspace(0, n - max_len, max_chunks).round().astype(int)
        return [(t[s:s + max_len], int(s)) for s in starts]
    raise ValueError(view)


def micro_batches(chunks, order, rows, seq):
    """Group documents (in the given order) into micro-batches whose chunks first-fit into rows x seq tokens.
    A document is never split across micro-batches. -> list of lists of (doc index, [(row, offset) per chunk])."""
    out, cur, free = [], [], [seq] * rows
    for d in order:
        place, trial = [], list(free)
        for c, _ in chunks[d]:
            r = next((i for i in range(rows) if trial[i] >= len(c)), None)
            if r is None:
                break
            place.append((r, seq - trial[r]))
            trial[r] -= len(c)
        if len(place) < len(chunks[d]):
            assert cur, f"document {d} does not fit into {rows} x {seq}"
            out.append(cur)
            cur, free = [], [seq] * rows
            trial, place = list(free), []
            for c, _ in chunks[d]:
                r = next(i for i in range(rows) if trial[i] >= len(c))
                place.append((r, seq - trial[r]))
                trial[r] -= len(c)
        cur.append((d, place))
        free = trial
    if cur:
        out.append(cur)
    return out


def pack(mb, chunks, doc_len, rows, seq, n_pos, device):
    """One micro-batch -> device tensors for DocClassifier."""
    ids = torch.zeros(rows, seq, dtype=torch.long)
    pos = torch.zeros(rows, seq, dtype=torch.long)
    seg = torch.full((rows, seq), PAD_SEG, dtype=torch.long)
    tok_chunk = torch.full((rows, seq), -1, dtype=torch.long)
    K = max(len(p) for _, p in mb)
    doc_slots = torch.full((len(mb), K), -1, dtype=torch.long)
    chunk_doc, chunk_pos, ci = [], [], 0
    for j, (d, place) in enumerate(mb):
        for k, ((c, start), (r, o)) in enumerate(zip(chunks[d], place)):
            L = len(c)
            ids[r, o:o + L] = torch.from_numpy(c)
            pos[r, o:o + L] = torch.arange(L)
            seg[r, o:o + L] = ci
            tok_chunk[r, o:o + L] = ci
            doc_slots[j, k] = ci
            chunk_doc.append(j)
            chunk_pos.append(min(n_pos - 1, int(n_pos * start / max(1, doc_len[d]))))
            ci += 1
    t = lambda x: x.to(device, non_blocking=True)  # noqa: E731
    return {"ids": t(ids), "pos": t(pos), "seg": t(seg), "tok_chunk": t(tok_chunk), "doc_slots": t(doc_slots),
            "chunk_doc": t(torch.tensor(chunk_doc)), "chunk_pos": t(torch.tensor(chunk_pos)),
            "docs": [d for d, _ in mb]}
