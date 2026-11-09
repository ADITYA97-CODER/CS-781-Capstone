"""BM25 parameter sweep with cached count matrices (per split, n-gram range, min_df). Prints F1@7 for each setting."""

import argparse
import itertools
import os

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

import pcr_data as D
from lexical import STOP, clean, OUT


def counts(split, ngram, min_df, numbers=False):
    path = os.path.join(OUT, "cache", f"{split}_ng{ngram[0]}{ngram[1]}_df{min_df}{'_num' if numbers else ''}.npz")
    qs, gold, pool = D.load(split)
    qids, cids = list(qs), list(pool)
    if os.path.exists(path):
        z = np.load(path)
        C = sp.csr_matrix((z["cd"], z["ci"], z["cp"]), shape=tuple(z["cs"]))
        Q = sp.csr_matrix((z["qd"], z["qi"], z["qp"]), shape=tuple(z["qs"]))
        return qids, cids, gold, Q, C
    vec = CountVectorizer(ngram_range=ngram, stop_words=sorted(STOP), min_df=min_df,
                          token_pattern=r"(?u)\b[a-z0-9][a-z0-9]+\b", dtype=np.float32)
    C = vec.fit_transform([clean(pool[c], numbers) for c in cids]).tocsr()
    Q = vec.transform([clean(qs[q], numbers) for q in qids]).tocsr()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, cd=C.data, ci=C.indices, cp=C.indptr, cs=C.shape, qd=Q.data, qi=Q.indices, qp=Q.indptr, qs=Q.shape)
    return qids, cids, gold, Q, C


def bm25(Q, C, k1, b, qpow=1.0, idf_floor=0.0):
    N = C.shape[0]
    df = np.asarray((C > 0).sum(0)).ravel()
    idf = np.log1p((N - df + 0.5) / (df + 0.5)).astype(np.float32)
    idf = np.where(idf < idf_floor, 0, idf)
    dl = np.asarray(C.sum(1)).ravel()
    Cc = C.tocoo()
    W = sp.csr_matrix(((Cc.data * (k1 + 1) / (Cc.data + k1 * (1 - b + b * dl[Cc.row] / dl.mean()))) * idf[Cc.col],
                       (Cc.row, Cc.col)), shape=C.shape)
    Qw = Q.copy()
    Qw.data = Qw.data ** qpow
    return np.asarray((Qw @ W.T).todense())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["dev", "train"])
    ap.add_argument("--ngram", type=int, nargs=2, default=[2, 2])
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--k1", type=float, nargs="+", default=[1.2, 2, 3, 5, 8])
    ap.add_argument("--b", type=float, nargs="+", default=[0.5, 0.75, 0.9, 1.0])
    ap.add_argument("--qpow", type=float, nargs="+", default=[1.0])
    ap.add_argument("--numbers", action="store_true")
    a = ap.parse_args()
    data = {s: counts(s, tuple(a.ngram), a.min_df, a.numbers) for s in a.splits}
    for k1, b, qp in itertools.product(a.k1, a.b, a.qpow):
        row = []
        for s in a.splits:
            qids, cids, gold, Q, C = data[s]
            row.append(D.score(gold, D.rank_from_scores(qids, cids, bm25(Q, C, k1, b, qp)))["f1"])
        print(f"ng {a.ngram} df {a.min_df} k1 {k1} b {b} qpow {qp} | " + " ".join(f"{s} {v:.4f}" for s, v in zip(a.splits, row)),
              flush=True)


if __name__ == "__main__":
    main()
