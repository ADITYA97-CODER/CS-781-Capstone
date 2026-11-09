"""Lexical retrieval baselines for PCR: BM25 / TF-IDF cosine over word n-grams (no learning, no external data).

IDF is computed on each split's own candidate pool (the documents being ranked).
  python lexical.py --split dev --ngram 1 2 --k1 1.2 --b 0.75
Saves the full score matrix to runs_p7/lex/<name>_<split>.npz (qids, cids, S) for fusion.
"""

import argparse
import os
import re
import time

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

import pcr_data as D

OUT = os.path.join(D.ROOT, "runs_p7", "lex")
_PLACE = re.compile(r"<[A-Z_]+>")
STOP = set(ENGLISH_STOP_WORDS) | {"name", "org", "said", "court", "section", "act", "case", "appellant", "respondent",
                                  "learned", "counsel", "high", "j", "v", "vs", "ors", "anr", "also"}


def clean(t, keep_numbers=False):
    t = _PLACE.sub(" ", t).lower()
    t = re.sub(r"[^a-z0-9]+" if keep_numbers else r"[^a-z]+", " ", t)
    return t


def bm25_matrix(q_texts, c_texts, ngram=(1, 1), k1=1.2, b=0.75, min_df=1, max_df=1.0, keep_numbers=False,
                query_tf="binary"):
    vec = CountVectorizer(ngram_range=ngram, stop_words=sorted(STOP), min_df=min_df, max_df=max_df,
                          token_pattern=r"(?u)\b[a-z0-9][a-z0-9]+\b", dtype=np.float32)
    C = vec.fit_transform([clean(t, keep_numbers) for t in c_texts]).tocsr()
    Q = vec.transform([clean(t, keep_numbers) for t in q_texts]).tocsr()
    N = C.shape[0]
    df = np.asarray((C > 0).sum(0)).ravel()
    idf = np.log1p((N - df + 0.5) / (df + 0.5)).astype(np.float32)
    dl = np.asarray(C.sum(1)).ravel()
    avg = dl.mean()
    C = C.tocoo()
    denom = C.data + k1 * (1 - b + b * dl[C.row] / avg)
    W = sp.csr_matrix(((C.data * (k1 + 1) / denom) * idf[C.col], (C.row, C.col)), shape=C.shape)
    if query_tf == "binary":
        Q.data[:] = 1.0
    elif query_tf == "log":
        Q.data = 1 + np.log(Q.data)
    return np.asarray((Q @ W.T).todense()), vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--ngram", type=int, nargs=2, default=[1, 1])
    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--min_df", type=int, default=1)
    ap.add_argument("--max_df", type=float, default=1.0)
    ap.add_argument("--query_tf", default="binary", choices=["binary", "raw", "log"])
    ap.add_argument("--numbers", action="store_true")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    t0 = time.time()
    qs, gold, pool = D.load(a.split)
    qids, cids = list(qs), list(pool)
    S, vec = bm25_matrix([qs[q] for q in qids], [pool[c] for c in cids], tuple(a.ngram), a.k1, a.b, a.min_df,
                         a.max_df, a.numbers, a.query_tf)
    msg = f"{a.split} bm25 ngram {a.ngram} k1 {a.k1} b {a.b} qtf {a.query_tf} nums {a.numbers} vocab {len(vec.vocabulary_)}"
    if a.split != "test":
        m = D.score(gold, D.rank_from_scores(qids, cids, S))
        msg += f" | F1@7 {m['f1']:.4f} P {m['p']:.4f} R {m['r']:.4f}"
    print(msg, f"({time.time() - t0:.0f}s)", flush=True)
    if a.name:
        os.makedirs(OUT, exist_ok=True)
        np.savez(os.path.join(OUT, f"{a.name}_{a.split}.npz"), qids=qids, cids=cids, S=S.astype(np.float32))


if __name__ == "__main__":
    main()
