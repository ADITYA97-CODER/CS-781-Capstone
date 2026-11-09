"""Penalty-aware MBR: BERTScore consensus minus reference-free penalties for known failure modes.

  score(c) = mean BERTScore F1 of c vs the other candidates
             - w_header  * [c contains a court header]           (ran into a new document)
             - w_close   * [c contains a closing formula]        (ended the judgment early)
             - w_sign    * [c contains a judge-signature block]  (ended the judgment early)
             - w_invent  * (# regex specifics in c not in the prompt)  (citations/dates/amounts/sections ...)
  Candidates with fewer than 128 words are only chosen if no 128-word candidate exists (as mbr_select.py).
  All penalties look only at the candidate and the prompt, never at a reference.

  # tune on even-index local prompts, report on odd-index ones and on all 990
  python phase1/mbr/mbr_rerank.py --cands outputs/phase1/mbr_archE/local_s{1..8}/prediction.json \
      --prompts local --tune --out outputs/phase1/mbr_rerank/archE_local.json
  # apply fixed weights (e.g. to the official prompts)
  python phase1/mbr/mbr_rerank.py --cands ... --prompts official --weights '{"header":.., "close":.., "sign":.., "invent":..}' --out ...
"""
import argparse
import hashlib
import itertools
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
from specifics import regex_specifics  # noqa: E402

PROMPTS = {"local": os.path.join(PROJECT_DIR, "data", "corpus", "eval", "local_test.json"),
           "official": os.path.join(PROJECT_DIR, "data", "Phase_1_Pretraining", "Test", "test.json")}
LOCAL_REF = os.path.join(PROJECT_DIR, "data", "corpus", "eval", "local_ref.json")
CACHE = os.path.join(PROJECT_DIR, "outputs", "phase1", "mbr_rerank", "cache")
HEADERS = re.compile(r"IN THE (SUPREME|HIGH) COURT|\bREPORTABLE\b|(CIVIL|CRIMINAL) (APPELLATE|ORIGINAL) JURISDICTION")
CLOSERS = re.compile(r"\b(appeal is (dis|al)lowed|petition is (dis|al)lowed|appeal dismissed|appeal allowed|"
                     r"no order as to costs|rule (is )?(made absolute|discharged)|parties to bear|"
                     r"stands? disposed of|bail bonds)", re.I)
# "J. (T.S. THAKUR)", "(JUDGE)", "Judge. Dated", "New Delhi, July 6, 2008", "Whether reporters of the local papers"
SIGN = re.compile(r"\bJ\.?\s*\(\s*[A-Z][A-Z.\s]+\)|\(\s*JUDGE\s*\)|\bNew Delhi[,;.]?\s+(dated\s+)?[A-Z][a-z]+\s+\d{1,2},?\s+\d{4}"
                  r"|Whether (the )?reporters of (the )?local papers", re.I)
DEFAULT_W = {"header": 0.0, "close": 0.0, "sign": 0.0, "invent": 0.0}
GRID = {"header": [0, 0.01, 0.03], "close": [0, 0.005, 0.01, 0.02], "sign": [0, 0.01, 0.03], "invent": [0, 0.001, 0.002, 0.004]}


def pair_scores(cands, ids, batch_size):
    """(n_ids, N, N) symmetric BERTScore F1 between candidates (cached on disk)."""
    key = hashlib.md5("|".join(f"{os.path.abspath(p)}:{os.path.getmtime(p)}" for p in cands).encode()
                      + str(len(ids)).encode()).hexdigest()
    path = os.path.join(CACHE, key + ".npy")
    if os.path.exists(path):
        return np.load(path)
    import bert_score
    data = [json.load(open(p)) for p in cands]
    N = len(cands)
    text = lambda c, i: data[c][i]["text"].strip() or "."
    pairs = list(itertools.combinations(range(N), 2))
    scorer = bert_score.BERTScorer(lang="en", batch_size=batch_size, device="cuda")
    _, _, f = scorer.score([text(x, i) for i in ids for x, _ in pairs], [text(y, i) for i in ids for _, y in pairs])
    f = np.array(f.tolist()).reshape(len(ids), len(pairs))
    S = np.zeros((len(ids), N, N))
    for j, (x, y) in enumerate(pairs):
        S[:, x, y] = S[:, y, x] = f[:, j]
    os.makedirs(CACHE, exist_ok=True)
    np.save(path, S)
    return S


def features(data, prompts, ids):
    """(n_ids, N, 5): header, close, sign, invented regex specifics, short."""
    N = len(data)
    F = np.zeros((len(ids), N, 5))
    for a, i in enumerate(ids):
        p = prompts[i]["text"]
        p_keys = set(regex_specifics(p))
        for c in range(N):
            t = data[c][i]["text"]
            F[a, c] = [bool(HEADERS.search(t)), bool(CLOSERS.search(t)), bool(SIGN.search(t)),
                       len(set(regex_specifics(t)) - p_keys), len(t.split()) < 128]
    return F


def select(S, F, w, loop_reps=None, loop_threshold=0.25):
    """loop_reps (n_ids, N) given = loop filter as in mbr_select.py: candidates with repetition > threshold are
    dropped and the consensus is computed among the rest; if every candidate loops, the least repetitive wins."""
    N = S.shape[1]
    if loop_reps is None:
        cons = S.sum(2) / (N - 1)
    else:
        keep = loop_reps <= loop_threshold                                 # (n_ids, N)
        cons = (S * keep[:, None, :]).sum(2) / np.maximum(keep.sum(1, keepdims=True) - 1, 1)
    score = cons - w["header"] * F[..., 0] - w["close"] * F[..., 1] - w["sign"] * F[..., 2] - w["invent"] * F[..., 3]
    score = np.where(F[..., 4] > 0, score - 10.0, score)          # prefer 128-word candidates
    if loop_reps is None:
        return score.argmax(1)
    score = np.where(keep, score, -np.inf)
    pick = score.argmax(1)
    all_loop = ~keep.any(1)
    pick[all_loop] = loop_reps[all_loop].argmin(1)
    return pick


def official_bertscore(cands_txt, refs_txt, scorer):
    _, _, f = scorer.score([c.strip() or "." for c in cands_txt], [r.strip() or "." for r in refs_txt])
    return np.array(f.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--prompts", default="local")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=None, help="json dict; default: all 0 (= plain MBR)")
    ap.add_argument("--tune", action="store_true", help="local only: grid-search weights on even-index prompts")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--loop-filter", action="store_true", help="drop looping candidates first (as mbr_select.py)")
    a = ap.parse_args()

    prompts = json.load(open(PROMPTS.get(a.prompts, a.prompts)))
    data = [json.load(open(p)) for p in a.cands]
    ids = [k for k in prompts if k in data[0]]
    S = pair_scores(a.cands, ids, a.batch_size)
    F = features(data, prompts, ids)
    w = dict(DEFAULT_W, **json.loads(a.weights)) if a.weights else dict(DEFAULT_W)
    LR = None
    if a.loop_filter:
        from mbr_select import repetition
        LR = np.array([[repetition(data[c][i]["text"]) for c in range(len(data))] for i in ids])

    if a.tune:
        import bert_score
        refs = json.load(open(LOCAL_REF))
        scorer = bert_score.BERTScorer(lang="en", batch_size=a.batch_size, device="cuda")
        # BERTScore of every candidate vs the reference (to score any selection without re-running the scorer)
        R = np.stack([official_bertscore([data[c][i]["text"] for i in ids], [refs[i]["text"] for i in ids], scorer)
                      for c in range(len(data))], 1)                           # (n_ids, N)
        dev = np.arange(len(ids)) % 2 == 0
        rows = []
        for vals in itertools.product(*GRID.values()):
            ww = dict(zip(GRID, vals))
            pick = select(S, F, ww)
            got = R[np.arange(len(ids)), pick]
            rows.append((got[dev].mean(), ww))
        rows.sort(key=lambda r: -r[0])
        w = rows[0][1]
        base_pick, pick = select(S, F, DEFAULT_W), select(S, F, w)
        base, new = R[np.arange(len(ids)), base_pick], R[np.arange(len(ids)), pick]
        rng = np.random.default_rng(0)
        report = {"weights": w, "top5_dev": [(float(s), ww) for s, ww in rows[:5]]}
        for name, sel in [("dev (even, tuned on)", dev), ("test (odd, held out)", ~dev), ("all 990", np.ones_like(dev))]:
            d = new[sel] - base[sel]
            bi = rng.integers(0, sel.sum(), size=(2000, sel.sum()))
            ci = np.percentile(d[bi].mean(1), [2.5, 97.5])
            report[name] = dict(mbr=float(base[sel].mean()), rerank=float(new[sel].mean()), delta=float(d.mean()),
                                ci=ci.tolist(), changed=int((pick != base_pick)[sel].sum()))
            print(f"{name:22s} plain MBR {base[sel].mean():.4f}  rerank {new[sel].mean():.4f}  Δ {d.mean():+.4f} "
                  f"({ci[0]:+.4f}…{ci[1]:+.4f})  picks changed {(pick != base_pick)[sel].sum()}")
        for nm, j in [("header", 0), ("close", 1), ("sign", 2)]:
            report[f"n_{nm}"] = [int(F[np.arange(len(ids)), base_pick, j].sum()), int(F[np.arange(len(ids)), pick, j].sum())]
        report["invented_regex_per_output"] = [float(F[np.arange(len(ids)), base_pick, 3].mean()),
                                               float(F[np.arange(len(ids)), pick, 3].mean())]
        print("weights", w, "| header/close/sign/invented: plain MBR -> rerank",
              report["n_header"], report["n_close"], report["n_sign"], report["invented_regex_per_output"])
        json.dump(report, open(os.path.splitext(a.out)[0] + "_tuning.json", "w"), indent=1)

    pick = select(S, F, w, LR)
    out = {i: {"text": data[pick[n]][i]["text"]} for n, i in enumerate(ids)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", a.out, "weights", w)


if __name__ == "__main__":
    main()
