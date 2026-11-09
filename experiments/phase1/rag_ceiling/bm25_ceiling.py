"""RAG ceiling for failure #1 (missing specifics): BM25 over the training corpus, measured on the
990 local held-out prompts.

Index: every training document cut into 256-token chunks (legal_bpe_32k token ids are the terms;
special tokens ignored). Scoring runs on GPU 0 as sparse(chunk x term, BM25-saturated tf) @
dense(term x query, idf). Two query variants per prompt: its last 32 words, and the whole prompt.

Specifics = capitalised words / words with digits in the reference that do NOT appear in the prompt.
Coverage@k = share of an example's specifics that appear in its top-k retrieved chunks.
Control: k random chunks (chance level). Chunks from near-duplicate documents of the held-out example
(from dup_check.json) can be excluded.

Output: outputs/phase1/rag_ceiling/bm25_topk.npz (retrieved chunk ids) and a printed report.
"""
import json
import os
import re
import time

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOK_DIR = os.path.join(PROJECT_DIR, "data", "tokens", "legal_bpe_32k")
OUT_DIR = os.path.join(PROJECT_DIR, "outputs", "phase1", "rag_ceiling")
SOURCES = ["supreme_court", "high_courts", "tribunals", "acts"]
CHUNK, V, N_SPECIAL = 256, 32000, 105
K1, B = 1.2, 0.75
TOPK, BATCH_CHUNKS = 50, 32768
KS = [1, 5, 10, 20]
ENT = re.compile(r"^[A-Z][A-Za-z.&'-]+$|\d")
STRIP = ",.;:()'\"[]"


def ents(text):
    return {w.strip(STRIP) for w in text.split() if ENT.search(w.strip(STRIP))} - {""}


def words(text):
    return {w.strip(STRIP) for w in text.split()} - {""}


def chunk_views():
    """Per source: memmap of the tokens reshaped to (n_chunks, CHUNK) and the doc start offsets."""
    out = []
    for src in SOURCES:
        t = np.memmap(os.path.join(TOK_DIR, f"{src}.train.bin"), dtype=np.uint16, mode="r")
        n = len(t) // CHUNK
        out.append((src, t[:n * CHUNK].reshape(n, CHUNK), np.load(os.path.join(TOK_DIR, f"{src}.train.idx.npy"))))
    return out


def saturated_tf(chunks):
    """Sparse CSR (n x V) of BM25-saturated term frequencies for a (n, CHUNK) uint16 array."""
    x = torch.from_numpy(chunks.astype(np.int64)).cuda()
    n = x.size(0)
    valid = x >= N_SPECIAL
    dl = valid.sum(1).float()
    rows = torch.arange(n, device="cuda")[:, None].expand_as(x)[valid]
    cols = x[valid]
    tf = torch.sparse_coo_tensor(torch.stack([rows, cols]), torch.ones_like(cols, dtype=torch.float32),
                                 (n, V)).coalesce()
    r, c = tf.indices()
    v = tf.values()
    norm = K1 * (1 - B + B * dl[r] / CHUNK)
    return torch.sparse_coo_tensor(tf.indices(), v * (K1 + 1) / (v + norm), (n, V)).to_sparse_csr(), (r, c)


def main():
    tok = PreTrainedTokenizerFast.from_pretrained(os.path.join(PROJECT_DIR, "artifacts", "tokenizer", "legal_bpe_32k"))
    prompts = json.load(open(os.path.join(PROJECT_DIR, "data", "corpus", "eval", "local_test.json")))
    refs = json.load(open(os.path.join(PROJECT_DIR, "data", "corpus", "eval", "local_ref.json")))
    ids = list(prompts)
    views = chunk_views()
    offsets = np.cumsum([0] + [v.shape[0] for _, v, _ in views])
    n_chunks = int(offsets[-1])
    print(f"{n_chunks:,} chunks of {CHUNK} tokens", flush=True)

    # --- document frequency from a 10% strided sample of chunks
    t0 = time.time()
    df, sampled = torch.zeros(V, device="cuda"), 0
    for src, v, _ in views:
        for s in range(0, v.shape[0], BATCH_CHUNKS * 10):
            batch = np.asarray(v[s:s + BATCH_CHUNKS])
            _, (r, c) = saturated_tf(batch)
            df += torch.bincount(c, minlength=V).float()
            sampled += batch.shape[0]
    idf = torch.log(1 + (sampled - df + 0.5) / (df + 0.5))
    idf[:N_SPECIAL] = 0
    print(f"df from {sampled:,} sampled chunks ({time.time() - t0:.0f}s)", flush=True)

    # --- query matrix: column q = idf of each distinct query term (two variants per prompt)
    def qtoks(text):
        return set(tok(" " + text, add_special_tokens=False)["input_ids"])
    queries = [qtoks(" ".join(prompts[k]["text"].split()[-32:])) for k in ids] + \
              [qtoks(prompts[k]["text"]) for k in ids]
    W = torch.zeros(V, len(queries), device="cuda")
    for q, ts in enumerate(queries):
        ts = torch.tensor(sorted(ts), device="cuda")
        W[ts, q] = idf[ts]

    # --- score every chunk, keep running top-TOPK per query
    best_s = torch.full((TOPK, len(queries)), -1.0, device="cuda")
    best_i = torch.zeros((TOPK, len(queries)), dtype=torch.long, device="cuda")
    t0 = time.time()
    for (src, v, _), base in zip(views, offsets[:-1]):
        for s in range(0, v.shape[0], BATCH_CHUNKS):
            tfm, _ = saturated_tf(np.asarray(v[s:s + BATCH_CHUNKS]))
            scores = (tfm @ W)                                          # (n, Q)
            k = min(TOPK, scores.size(0))
            bs, bi = scores.topk(k, dim=0)
            cat_s = torch.cat([best_s, bs]); cat_i = torch.cat([best_i, bi + int(base) + s])
            best_s, sel = cat_s.topk(TOPK, dim=0)
            best_i = cat_i.gather(0, sel)
        print(f"scored {src} ({time.time() - t0:.0f}s)", flush=True)
    best_i = best_i.T.cpu().numpy()                                     # (Q, TOPK), best first

    # --- chunk -> (source, doc id) and text
    doc_base = np.cumsum([0] + [len(ix) for _, _, ix in views])
    def locate(g):
        si = int(np.searchsorted(offsets, g, side="right") - 1)
        src, v, ix = views[si]
        local = g - offsets[si]
        return si, int(doc_base[si] + np.searchsorted(ix, local * CHUNK, side="right") - 1), v[local]
    uniq = np.unique(best_i)
    rng = np.random.default_rng(0)
    rand = rng.integers(0, n_chunks, size=(len(ids), max(KS)))
    all_needed = np.unique(np.concatenate([uniq, rand.ravel()]))
    info = {}
    for g in all_needed:
        si, d, toks = locate(int(g))
        info[int(g)] = (si, d, words(tok.decode([t for t in toks.tolist() if t >= N_SPECIAL])))
    np.savez(os.path.join(OUT_DIR, "bm25_topk.npz"), best_i=best_i, rand=rand)

    # --- coverage report
    while not os.path.exists(os.path.join(OUT_DIR, "dup_check.json")):     # written by dup_check.py
        time.sleep(30)
    dup = json.load(open(os.path.join(OUT_DIR, "dup_check.json")))["parts"]
    near = {k: set(dup[f"local_prompt:{k}"]["near_dup_docs"]) | set(dup[f"local_ref:{k}"]["near_dup_docs"]) for k in ids}
    spec = {k: ents(refs[k]["text"]) - ents(prompts[k]["text"]) for k in ids}
    src_of = {k: refs[k]["source"] for k in ids}

    def coverage(ranked, kk, exclude):
        per, per_src, tot_hit, tot = [], {}, 0, 0
        for n, k in enumerate(ids):
            if not spec[k]:
                continue
            chunks = [g for g in ranked[n] if not (exclude and info[int(g)][1] in near[k])][:kk]
            got = set().union(*(info[int(g)][2] for g in chunks)) if chunks else set()
            hit = len(spec[k] & got)
            per.append(hit / len(spec[k])); per_src.setdefault(src_of[k], []).append(hit / len(spec[k]))
            tot_hit += hit; tot += len(spec[k])
        return np.mean(per), tot_hit / tot, {s: np.mean(v) for s, v in per_src.items()}

    Q = len(ids)
    print(f"\nexamples with >=1 new specific: {sum(bool(spec[k]) for k in ids)} / {Q}; "
          f"mean new specifics per example: {np.mean([len(spec[k]) for k in ids]):.1f}")
    print(f"examples with a near-duplicate training doc (>= 30% shared 12-grams): {sum(bool(near[k]) for k in ids)}")
    header = "setting".ljust(34) + "".join(f"cov@{k}".rjust(9) for k in KS) + "   @10 by source (SC / HC / Trib / Acts)"
    print(header)
    rows = [("random chunks (chance)", rand, False), ("BM25 last-32-words", best_i[:Q], False),
            ("BM25 last-32-words, no near-dups", best_i[:Q], True), ("BM25 full prompt", best_i[Q:], False),
            ("BM25 full prompt, no near-dups", best_i[Q:], True)]
    for name, ranked, ex in rows:
        cells, bysrc = [], None
        for kk in KS:
            m, micro, bs = coverage(ranked, kk, ex)
            cells.append(f"{m:9.3f}")
            if kk == 10:
                bysrc = bs
        print(name.ljust(34) + "".join(cells) + "   " + " / ".join(f"{bysrc.get(s, float('nan')):.3f}" for s in
                                                                 ["supreme_court", "high_courts", "tribunals", "acts"]))


if __name__ == "__main__":
    main()
