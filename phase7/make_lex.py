"""Save lexical score matrices for fusion / hard negatives: runs_p7/lex/<name>_<split>.npz (qids, cids, S).
  python make_lex.py --name bm25bi --ngram 2 2 --min_df 2 --k1 4 --b 0.95"""
import argparse
import os

import numpy as np

from lex_sweep import counts, bm25
from lexical import OUT

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--ngram", type=int, nargs=2, default=[2, 2])
ap.add_argument("--min_df", type=int, default=2)
ap.add_argument("--k1", type=float, default=4)
ap.add_argument("--b", type=float, default=0.95)
ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
a = ap.parse_args()
for s in a.splits:
    qids, cids, gold, Q, C = counts(s, tuple(a.ngram), a.min_df)
    np.savez(os.path.join(OUT, f"{a.name}_{s}.npz"), qids=qids, cids=cids, S=bm25(Q, C, a.k1, a.b).astype(np.float32))
    print(a.name, s, "saved", flush=True)
