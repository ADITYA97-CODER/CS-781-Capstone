"""Token cache for PCR documents (legal_bpe_32k). Every query is also in its split's pool with identical text, so
only the pool is tokenized: runs_p7/cache/<split>.npz (ids, lens, flat uint16)."""

import os

import numpy as np
from tokenizers import Tokenizer

import pcr_data as D

CACHE = os.path.join(D.ROOT, "runs_p7", "cache")
TOKENIZER = os.path.join(D.ROOT, "artifacts", "tokenizer", "legal_bpe_32k", "tokenizer.json")


def load_tokens(split):
    """-> (doc ids list, {id: np.int32 token array})."""
    path = os.path.join(CACHE, f"{split}.npz")
    if not os.path.exists(path):
        qs, _, pool = D.load(split)
        assert all(q in pool for q in qs)
        ids = list(pool)
        tok = Tokenizer.from_file(TOKENIZER)
        enc = tok.encode_batch([pool[i] for i in ids])
        lens = np.array([len(e.ids) for e in enc], dtype=np.int64)
        flat = np.concatenate([np.asarray(e.ids, dtype=np.uint16) for e in enc])
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, ids=np.array(ids), lens=lens, flat=flat)
    z = np.load(path)
    offs = np.concatenate([[0], np.cumsum(z["lens"])])
    flat = z["flat"].astype(np.int32)
    ids = [str(i) for i in z["ids"]]
    return ids, {d: flat[offs[i]:offs[i + 1]] for i, d in enumerate(ids)}


if __name__ == "__main__":
    for s in ("train", "dev", "test"):
        ids, t = load_tokens(s)
        L = np.array([len(v) for v in t.values()])
        print(s, len(ids), "docs, tokens median", int(np.median(L)), "p90", int(np.percentile(L, 90)), "total", L.sum())
