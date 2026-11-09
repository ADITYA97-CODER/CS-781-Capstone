"""Follow-up: near-duplicate breakdown and BM25 coverage restricted to rare specifics."""
import json, os, re
import numpy as np
from transformers import PreTrainedTokenizerFast
import sys; sys.path.insert(0, os.path.dirname(__file__))
from bm25_ceiling import ents, words, chunk_views, CHUNK, N_SPECIAL, PROJECT_DIR, OUT_DIR

tok = PreTrainedTokenizerFast.from_pretrained(os.path.join(PROJECT_DIR, "artifacts", "tokenizer", "legal_bpe_32k"))
P = json.load(open(os.path.join(PROJECT_DIR, "data/corpus/eval/local_test.json")))
R = json.load(open(os.path.join(PROJECT_DIR, "data/corpus/eval/local_ref.json")))
ids = list(P); SRC = ["supreme_court", "high_courts", "tribunals", "acts"]
dup = json.load(open(os.path.join(OUT_DIR, "dup_check.json")))["parts"]

print("== Near-duplicates of held-out examples (best single training doc, share of non-boilerplate 12-grams)")
for s in SRC + ["ALL"]:
    ks = [k for k in ids if s == "ALL" or R[k]["source"] == s]
    pf = np.array([dup[f"local_prompt:{k}"]["max_frac"] for k in ks]); rf = np.array([dup[f"local_ref:{k}"]["max_frac"] for k in ks])
    nd = sum(bool(dup[f"local_prompt:{k}"]["near_dup_docs"] or dup[f"local_ref:{k}"]["near_dup_docs"]) for k in ks)
    print(f"{s:14s} n={len(ks):3d} near-dup(>=30%) {nd:3d} ({nd/len(ks):.0%}) | reference >=50% verbatim in one doc: "
          f"{(rf>=.5).sum():3d} ({(rf>=.5).mean():.0%}), >=90%: {(rf>=.9).sum():3d} | prompt >=50%: {(pf>=.5).sum():3d}")
of = np.array([v["max_frac"] for k, v in dup.items() if k.startswith("official_prompt:")])
print(f"OFFICIAL prompts n={len(of)}: >=30% in one training doc {(of>=.3).sum()} ({(of>=.3).mean():.0%}), "
      f">=50% {(of>=.5).sum()}, >=90% {(of>=.9).sum()}, median {np.median(of):.2f}")

# rare specifics: not found in a background sample of 3000 random chunks
views = chunk_views(); offsets = np.cumsum([0] + [v.shape[0] for _, v, _ in views])
doc_base = np.cumsum([0] + [len(ix) for _, _, ix in views])
def chunk(g):
    si = int(np.searchsorted(offsets, g, side="right") - 1); v, ix = views[si][1], views[si][2]; loc = g - offsets[si]
    return int(doc_base[si] + np.searchsorted(ix, loc * CHUNK, side="right") - 1), v[loc]
def text(t): return words(tok.decode([x for x in t.tolist() if x >= N_SPECIAL]))
rng = np.random.default_rng(1); bg = set()
for g in rng.integers(0, offsets[-1], 3000): bg |= text(chunk(int(g))[1])
z = np.load(os.path.join(OUT_DIR, "bm25_topk.npz")); best, rand = z["best_i"][:len(ids)], z["rand"]
cache = {}
def info(g):
    if g not in cache: d, t = chunk(g); cache[g] = (d, text(t))
    return cache[g]
near = {k: set(dup[f"local_prompt:{k}"]["near_dup_docs"]) | set(dup[f"local_ref:{k}"]["near_dup_docs"]) for k in ids}
spec = {k: ents(R[k]["text"]) - ents(P[k]["text"]) for k in ids}
rare = {k: {w for w in spec[k] if w not in bg} for k in ids}
print(f"\n== Rare specifics (absent from 3000 random chunks): {np.mean([len(rare[k]) for k in ids]):.1f} per example "
      f"(of {np.mean([len(spec[k]) for k in ids]):.1f}); examples with any: {sum(bool(rare[k]) for k in ids)}")
print("setting".ljust(36) + "".join(f"cov@{k}".rjust(8) for k in [1, 5, 10, 20]) + "   @10 SC / HC / Trib / Acts")
for name, ranked, ex in [("random chunks", rand, False), ("BM25 last-32", best, False), ("BM25 last-32, no near-dups", best, True)]:
    cells = []
    for kk in [1, 5, 10, 20]:
        per, bys = [], {}
        for n, k in enumerate(ids):
            if not rare[k]: continue
            ch = [int(g) for g in ranked[n] if not (ex and info(int(g))[0] in near[k])][:kk]
            got = set().union(*(info(g)[1] for g in ch)) if ch else set()
            c = len(rare[k] & got) / len(rare[k]); per.append(c); bys.setdefault(R[k]["source"], []).append(c)
        cells.append(f"{np.mean(per):8.3f}")
        if kk == 10: b = bys
    print(name.ljust(36) + "".join(cells) + "   " + " / ".join(f"{np.mean(b[s]):.3f}" for s in SRC))
