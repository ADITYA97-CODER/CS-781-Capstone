"""Per-split (query x pool) feature matrices for learning-to-rank / fusion. Cached in runs_p7/feat/<split>.npz.

Lexical (no learning): BM25 over word 1/2/3-grams (IDF on the split's pool), with two query views:
  full  the whole query text
  ctx   only the sentences around the query's <CITATION> placeholders (+-1 sentence): where the query cites prior cases
Metadata (parsed from the documents, pcr_data.meta):
  date_viol   candidate DATE OF JUDGMENT later than the query's (never true for a gold pair on train/dev)
  year_viol   candidate's latest mentioned year - query's latest mentioned year, clipped at 0 (proxy for undated docs)
  year_gap    query latest year - candidate latest year
  is_query    the candidate is itself one of the split's queries
  citator     the candidate's CITATOR INFO lists the query's AIR citation  (citation-graph signal left in the header)
  cite_text   the query text (after the header) contains the candidate's own AIR / SCR citation (an unmasked citation)
"""

import json
import os
import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import pcr_data as D
from lex_sweep import bm25
from lexical import STOP, clean

FEAT = os.path.join(D.ROOT, "runs_p7", "feat")
LEX_CFG = {"uni": ((1, 1), 10.0, 1.0), "bi": ((2, 2), 6.0, 1.0), "tri": ((3, 3), 10.0, 1.0)}


def ctx_text(sents, w=1):
    idx = [i for i, s in enumerate(sents) if "<CITATION>" in s]
    keep = sorted({j for i in idx for j in range(max(0, i - w), min(len(sents), i + w + 1))})
    return " ".join(sents[j] for j in keep) if keep else " ".join(sents)


def own_cite_patterns(t):
    m = re.search(r"CITATION:(.*?)(?:CITATOR INFO|ACT:|HEADNOTE:|JUDGMENT:)", t[:6000], re.S)
    if not m:
        return []
    s, out = m.group(1), []
    for y, p in re.findall(r"(\d{4}) AIR\s*(\d+)", s):
        out.append(rf"{y}\s*AIR\s*\(?S\.?C\.?\)?\s*{p}\b|AIR\s*{y}\s*SC\s*{p}\b")
    for y, v, p in re.findall(r"(\d{4}) SCR\s*\((\d)\)\s*(\d+)", s):
        out.append(rf"\[?{y}\]?\s*\(?{v}\)?\s*S\.?\s*C\.?\s*R\.?\s*{p}\b")
    for y, p in re.findall(r"(\d{4}) SCR\s+(\d+)", s):
        out.append(rf"\[?{y}\]?\s*S\.?\s*C\.?\s*R\.?\s*{p}\b")
    return [re.compile(p) for p in out]


def build(split):
    path = os.path.join(FEAT, f"{split}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return list(z["qids"]), list(z["cids"]), {k: z[k] for k in z.files if k not in ("qids", "cids")}
    a, b = D.FILES[split]
    q = json.load(open(os.path.join(D.DATA, a)))
    c = json.load(open(os.path.join(D.DATA, b)))
    qids, cids = list(q), list(c)
    full_q = [" ".join(q[k]["text"]) for k in qids]
    ctx_q = [ctx_text(q[k]["text"]) for k in qids]
    pool = [" ".join(c[k]["text"]) for k in cids]
    F = {}
    for name, (ng, k1, bb) in LEX_CFG.items():
        vec = CountVectorizer(ngram_range=ng, stop_words=sorted(STOP), min_df=2,
                              token_pattern=r"(?u)\b[a-z0-9][a-z0-9]+\b", dtype=np.float32)
        C = vec.fit_transform([clean(t) for t in pool]).tocsr()
        F[f"{name}_full"] = bm25(vec.transform([clean(t) for t in full_q]).tocsr(), C, k1, bb).astype(np.float32)
        F[f"{name}_ctx"] = bm25(vec.transform([clean(t) for t in ctx_q]).tocsr(), C, k1, bb).astype(np.float32)
        print(split, name, "done", flush=True)
    qm = [D.meta(t) for t in full_q]
    cm = [D.meta(t) for t in pool]
    nq, nc = len(qids), len(cids)
    date_viol = np.zeros((nq, nc), np.float32)
    citator = np.zeros((nq, nc), np.float32)
    qy = np.array([m["year_max"] or 0 for m in qm], np.float32)
    cy = np.array([m["year_max"] or 0 for m in cm], np.float32)
    has = (qy[:, None] > 0) & (cy[None] > 0)
    year_viol = np.where(has, np.clip(cy[None] - qy[:, None], 0, None), 0).astype(np.float32)
    year_gap = np.where(has, np.clip(qy[:, None] - cy[None], -50, 100), 0).astype(np.float32)
    for i, m in enumerate(qm):
        for j, n in enumerate(cm):
            if m["date"] and n["date"] and n["date"] > m["date"]:
                date_viol[i, j] = 1
            if m["cites"] and m["cites"] & n["citator"]:
                citator[i, j] = 1
    qset = set(qids)
    is_query = np.tile(np.array([x in qset for x in cids], np.float32), (nq, 1))
    cite_text = np.zeros((nq, nc), np.float32)
    pats = [own_cite_patterns(t) for t in pool]
    for i, t in enumerate(full_q):
        body = t[3000:]
        for j, ps in enumerate(pats):
            if ps and any(p.search(body) for p in ps):
                cite_text[i, j] = 1
    F.update(date_viol=date_viol, year_viol=year_viol, year_gap=year_gap, is_query=is_query, citator=citator,
             cite_text=cite_text)
    pos = {x: j for j, x in enumerate(cids)}
    for i, k in enumerate(qids):       # the query itself is never a candidate: zero its own column everywhere
        for v in F.values():
            v[i, pos[k]] = 0
    os.makedirs(FEAT, exist_ok=True)
    np.savez(path, qids=qids, cids=cids, **F)
    return qids, cids, F


if __name__ == "__main__":
    import sys
    for s in sys.argv[1:] or ["dev", "train", "test"]:
        qids, cids, F = build(s)
        if s != "test":
            _, gold, _ = D.load(s)
            for k, v in F.items():
                if k.endswith(("_full", "_ctx")):
                    print(s, k, round(D.score(gold, D.rank_from_scores(qids, cids, v))["f1"], 4))


TFIDF_CFG = {"tf13_full": ((1, 3), "full"), "tf13_ctx": ((1, 3), "ctx"), "tf22_ctx": ((2, 2), "ctx"),
             "tf33_full": ((3, 3), "full")}


def build_tfidf(split):
    """TF-IDF (sublinear tf, l2) cosine views, fitted on the split's pool. Cached in runs_p7/feat/<split>_tfidf.npz."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    path = os.path.join(FEAT, f"{split}_tfidf.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    a, b = D.FILES[split]
    q = json.load(open(os.path.join(D.DATA, a)))
    c = json.load(open(os.path.join(D.DATA, b)))
    qids, cids = list(q), list(c)
    views = {"full": [clean(" ".join(q[k]["text"])) for k in qids], "ctx": [clean(ctx_text(q[k]["text"])) for k in qids]}
    pool = [clean(" ".join(c[k]["text"])) for k in cids]
    out, fitted = {}, {}
    for name, (ng, view) in TFIDF_CFG.items():
        if ng not in fitted:
            v = TfidfVectorizer(ngram_range=ng, stop_words=sorted(STOP), min_df=2, sublinear_tf=True,
                                token_pattern=r"(?u)\b[a-z0-9][a-z0-9]+\b", dtype=np.float32)
            fitted[ng] = (v, v.fit_transform(pool))
        v, C = fitted[ng]
        out[name] = (v.transform(views[view]) @ C.T).toarray().astype(np.float32)
        print(split, name, "done", flush=True)
    pos = {x: j for j, x in enumerate(cids)}
    for i, k in enumerate(qids):
        for m in out.values():
            m[i, pos[k]] = 0
    np.savez(path, **out)
    return out
