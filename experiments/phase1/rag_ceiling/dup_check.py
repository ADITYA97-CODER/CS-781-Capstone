"""Near-duplicate check: how much of each held-out prompt / reference (and each official test prompt)
appears verbatim in a single training document.

Token 12-grams of the eval texts are hashed; every 12-gram position of the training corpus is hashed
the same way and matched. 12-grams that occur in more than MAX_DOCS training documents are treated
as boilerplate and ignored. For each eval example we report the largest fraction of its (non-boilerplate)
12-grams shared with one training document, and list the documents sharing >= NEAR_DUP_FRAC.

Output: outputs/phase1/rag_ceiling/dup_check.json
"""
import json
import os
import time

import numpy as np
from transformers import PreTrainedTokenizerFast

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOK_DIR = os.path.join(PROJECT_DIR, "data", "tokens", "legal_bpe_32k")
OUT_DIR = os.path.join(PROJECT_DIR, "outputs", "phase1", "rag_ceiling")
SOURCES = ["supreme_court", "high_courts", "tribunals", "acts"]
N, BASE = 12, np.uint64(1000003)
MAX_DOCS = 20          # 12-grams in more training docs than this are boilerplate
NEAR_DUP_FRAC = 0.3    # a training doc sharing >= 30% of an example's 12-grams is a near-duplicate
BATCH = 200_000_000


def ngram_hashes(tokens):
    """Polynomial hash of every length-N window (uint64 arithmetic wraps mod 2^64)."""
    t = tokens.astype(np.uint64)
    L = len(t) - N + 1
    if L <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(L, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(N):
            h = h * BASE + t[j:j + L]
    return h


def main():
    tok = PreTrainedTokenizerFast.from_pretrained(os.path.join(PROJECT_DIR, "artifacts", "tokenizer", "legal_bpe_32k"))
    enc = lambda s: np.array(tok(" " + s, add_special_tokens=False)["input_ids"], dtype=np.uint16)
    prompts = json.load(open(os.path.join(PROJECT_DIR, "data", "corpus", "eval", "local_test.json")))
    refs = json.load(open(os.path.join(PROJECT_DIR, "data", "corpus", "eval", "local_ref.json")))
    official = json.load(open(os.path.join(PROJECT_DIR, "data1", "Phase_1_Pretraining", "Test", "test.json")))

    # eval "parts": (name, set of hashes)
    parts = []
    for k in prompts:
        parts.append((f"local_prompt:{k}", np.unique(ngram_hashes(enc(prompts[k]["text"])))))
        parts.append((f"local_ref:{k}", np.unique(ngram_hashes(enc(refs[k]["text"])))))
    for k in official:
        parts.append((f"official_prompt:{k}", np.unique(ngram_hashes(enc(official[k]["text"])))))
    all_h = np.concatenate([h for _, h in parts])
    part_of = np.concatenate([np.full(len(h), i, dtype=np.int32) for i, (_, h) in enumerate(parts)])
    uniq, inv = np.unique(all_h, return_inverse=True)          # hash id = index into uniq
    print(f"{len(parts)} eval parts, {len(uniq):,} distinct 12-grams", flush=True)

    # stream the training corpus: collect (hash id, global doc id) pairs
    pairs, doc_base, t0 = [], 0, time.time()
    doc_ranges = {}
    for src in SOURCES:
        toks = np.memmap(os.path.join(TOK_DIR, f"{src}.train.bin"), dtype=np.uint16, mode="r")
        idx = np.load(os.path.join(TOK_DIR, f"{src}.train.idx.npy"))
        doc_ranges[src] = (doc_base, doc_base + len(idx))
        for s in range(0, len(toks), BATCH):
            e = min(len(toks), s + BATCH + N - 1)
            h = ngram_hashes(np.asarray(toks[s:e]))
            pos = np.searchsorted(uniq, h)
            pos[pos == len(uniq)] = 0
            hit = uniq[pos] == h
            hid = pos[hit]
            doc = doc_base + np.searchsorted(idx, s + np.nonzero(hit)[0], side="right") - 1
            pairs.append(np.unique(hid.astype(np.int64) * 10_000_000 + doc))
            print(f"{src} {e / len(toks):.0%} ({time.time() - t0:.0f}s)", flush=True)
        doc_base += len(idx)
    keys = np.unique(np.concatenate(pairs))
    hid, doc = keys // 10_000_000, keys % 10_000_000
    ndocs = np.bincount(hid, minlength=len(uniq))               # training docs containing each 12-gram
    common = ndocs > MAX_DOCS
    print(f"boilerplate 12-grams (> {MAX_DOCS} docs): {common.sum():,} of {(ndocs > 0).sum():,} matched", flush=True)

    # per eval part: best single-doc overlap over non-boilerplate 12-grams
    keep = ~common[hid]
    hid, doc = hid[keep], doc[keep]
    order = np.argsort(hid, kind="stable")
    hid, doc = hid[order], doc[order]
    starts = np.searchsorted(hid, np.arange(len(uniq)))
    ends = np.searchsorted(hid, np.arange(len(uniq)), side="right")
    part_hashes = {}
    for h_id, p in zip(inv, part_of):
        part_hashes.setdefault(p, []).append(h_id)
    result = {}
    for p, (name, _) in enumerate(parts):
        hs = np.unique(np.array(part_hashes.get(p, []), dtype=np.int64))
        hs = hs[~common[hs]]
        if len(hs) == 0:
            result[name] = {"max_frac": 0.0, "near_dup_docs": [], "n_grams": 0}
            continue
        docs = np.concatenate([doc[starts[h]:ends[h]] for h in hs]) if len(hs) else np.zeros(0, np.int64)
        if len(docs) == 0:
            result[name] = {"max_frac": 0.0, "near_dup_docs": [], "n_grams": int(len(hs))}
            continue
        d, c = np.unique(docs, return_counts=True)
        frac = c / len(hs)
        result[name] = {"max_frac": float(frac.max()), "n_grams": int(len(hs)),
                        "near_dup_docs": [int(x) for x in d[frac >= NEAR_DUP_FRAC]]}
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump({"doc_ranges": doc_ranges, "parts": result}, open(os.path.join(OUT_DIR, "dup_check.json"), "w"))
    print("wrote", os.path.join(OUT_DIR, "dup_check.json"))


if __name__ == "__main__":
    main()
