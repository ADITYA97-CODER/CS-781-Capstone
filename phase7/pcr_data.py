"""Phase 7 Prior Case Retrieval: data loading, metadata and the official metric.

Splits: train (827 queries / 4,320 candidates), dev (118 / 1,023), test (237 / 1,727). Each split has its own pool.
Every query also appears in its own pool with identical text (EXPERIMENTS.md P7.1) -> always excluded from its ranking.

Metadata parsed from the IndianKanoon header (present for ~25-40% of documents):
  date      "DATE OF JUDGMENT: dd/mm/yyyy"  (a gold candidate is never later than its query: 0 / 1,749 dated pairs)
  cites     own AIR citations "CITATION: 1964 AIR 1545"
  citator   later cases citing this one, "CITATOR INFO : R 1970 SC 765 (9) ..." -> (year, page) pairs
  year_max  the latest 4-digit year 1850..2025 mentioned anywhere in the text (a proxy date for every document)
"""

import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "Phase_7_PCR")
FILES = {"train": ("Train/train.json", "Train/train_candidates.json"),
         "dev": ("Dev/dev.json", "Dev/dev_candidates.json"),
         "test": ("Test/test.json", "Test/test_candidates.json")}
K = 7


def load(split):
    """-> (queries {id: text str}, gold {id: [cand ids]} (empty for test), pool {id: text str})."""
    a, b = FILES[split]
    q = json.load(open(os.path.join(DATA, a)))
    c = json.load(open(os.path.join(DATA, b)))
    queries = {k: " ".join(v["text"]) for k, v in q.items()}
    gold = {k: [x for x in v.get("relevant_candidates", []) if x.strip()] for k, v in q.items()}
    pool = {k: " ".join(v["text"]) for k, v in c.items()}
    return queries, gold, pool


_DATE = re.compile(r"DATE OF JUDGMENT:?\s*(\d\d)/(\d\d)/(\d{4})")
_CIT = re.compile(r"CITATION:(.*?)(?:CITATOR INFO|ACT:|HEADNOTE:|JUDGMENT:|$)", re.S)
_CITATOR = re.compile(r"CITATOR INFO\s*:(.*?)(?:ACT:|HEADNOTE:|JUDGMENT:|$)", re.S)
_YEAR = re.compile(r"(?<!\d)(1[89]\d\d|20[0-2]\d)(?!\d)")


def meta(text):
    head = text[:6000]
    m = _DATE.search(head)
    date = (int(m.group(3)), int(m.group(2)), int(m.group(1))) if m else None
    m = _CIT.search(head)
    cites = set(re.findall(r"(\d{4}) AIR\s*(\d+)", m.group(1))) if m else set()
    m = _CITATOR.search(head)
    citator = set(re.findall(r"(\d{4}) SC\s*(\d+)", m.group(1))) if m else set()
    years = [int(y) for y in _YEAR.findall(text)]
    return {"date": date, "cites": cites, "citator": citator, "year_max": max(years) if years else None}


def score(gold, pred):
    """Official Micro-F1@7 (evaluate_phase7.py). pred {qid: ranked list}."""
    tp = fp = fn = 0
    for q, labels in gold.items():
        g, p = set(labels), set(pred[q][:K])
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return {"f1": f1, "p": tp / max(1, tp + fp), "r": tp / max(1, tp + fn)}


def rank_from_scores(qids, cids, S, exclude_self=True):
    """S: (n_q, n_c) score matrix -> {qid: top-K cand ids}, the query itself removed."""
    pos = {c: j for j, c in enumerate(cids)}
    S = np.array(S, dtype=np.float64, copy=True)
    out = {}
    for i, q in enumerate(qids):
        if exclude_self and q in pos:
            S[i, pos[q]] = -np.inf
        top = np.argsort(-S[i], kind="stable")[:K]
        out[q] = [cids[j] for j in top]
    return out


def write_submission(pred, pool_ids, query_ids, out_dir):
    import zipfile
    assert set(pred) == set(query_ids)
    pool_ids = set(pool_ids)
    for q, v in pred.items():
        assert len(v) == K and len(set(v)) == K and all(c in pool_ids for c in v) and q not in v, q
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "prediction.json")
    json.dump({q: list(pred[q]) for q in query_ids}, open(path, "w"), indent=1)
    with zipfile.ZipFile(os.path.join(out_dir, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, "prediction.json")
    return os.path.join(out_dir, "submission.zip")
