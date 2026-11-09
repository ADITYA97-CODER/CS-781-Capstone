"""Minimum-Bayes-risk selection: for each prompt keep the candidate with the highest mean
BERTScore F1 against the other candidates. Uses no references.

Usage:
  python phase1/mbr/mbr_select.py --cands a.json b.json c.json ... --out prediction.json
  ... --loop-filter      drop looping candidates (> 25% repeated word 4-grams) before selection; the
                         consensus is then computed among the remaining candidates only. If every
                         candidate loops, the one with the lowest repetition score is kept.
"""
import argparse
import itertools
import json
import os
from collections import Counter

import bert_score


def repetition(text, n=4):
    """Share of repeated word n-grams (the loop metric of the error analysis; > 0.25 = loop)."""
    w = text.split()
    g = Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    tot = sum(g.values())
    return (tot - len(g)) / tot if tot else 0.0


def choose(texts, pair_f, min_words=128, loop_filter=False, loop_threshold=0.25):
    """texts: candidate strings; pair_f: {(x, y): F1} for x < y. Returns (index, consensus of the pick)."""
    N = len(texts)
    pool = list(range(N))
    if loop_filter:
        reps = [repetition(t) for t in texts]
        pool = [c for c in pool if reps[c] <= loop_threshold] or [min(pool, key=reps.__getitem__)]
    tot = [0.0] * N
    for (x, y), s in pair_f.items():
        if x in pool and y in pool:
            tot[x] += s
            tot[y] += s
    full = [c for c in pool if len(texts[c].split()) >= min_words]
    best = max(full or pool, key=tot.__getitem__)
    return best, tot[best] / max(len(pool) - 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", nargs="+", required=True, help="prediction.json files, one per candidate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--min-words", type=int, default=128,
                    help="prefer candidates with at least this many words (task requires exactly 128)")
    ap.add_argument("--loop-filter", action="store_true", help="drop looping candidates before selection")
    ap.add_argument("--loop-threshold", type=float, default=0.25)
    args = ap.parse_args()

    cands = [json.load(open(p)) for p in args.cands]
    ids, N = list(cands[0]), len(cands)
    text = lambda c, i: cands[c][i]["text"].strip() or "."
    # same model / layer / no idf / no rescaling as the official bert_score.score(lang='en')
    scorer = bert_score.BERTScorer(lang="en", batch_size=args.batch_size, device="cuda")

    pairs = list(itertools.combinations(range(N), 2))    # F1 is symmetric, so score each pair once
    a = [text(x, i) for i in ids for x, _ in pairs]
    b = [text(y, i) for i in ids for _, y in pairs]
    _, _, f = scorer.score(a, b)
    f = f.tolist()

    out, picked, support = {}, Counter(), []
    for n, i in enumerate(ids):
        pair_f = dict(zip(pairs, f[n * len(pairs):(n + 1) * len(pairs)]))
        best, cons = choose([cands[c][i]["text"] for c in range(N)], pair_f, args.min_words,
                            args.loop_filter, args.loop_threshold)
        out[i] = {"text": cands[best][i]["text"]}
        picked[args.cands[best]] += 1
        support.append(cons)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("picked per candidate file:", dict(picked))
    print(f"mean consensus score of picks: {sum(support) / len(support):.4f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
