"""Learning-to-rank fusion for PCR: logistic regression over (query, candidate) features.

Per score matrix (lexical BM25 views, dense cosine of trained bi-encoders) three transforms are used:
  z     z-score within the query's row (scale-free across queries)
  cz    z-score of z within the candidate's column (how much this query wants c relative to the other queries:
        every non-query candidate of a pool is gold for >= 1 query of that split, so competition between queries
        is informative; a transductive normalisation that uses no labels)
  rr    1 / (rank within the row + 1)
plus the metadata flags (features.py). Pairs = top --cand per query by summed lexical z (positives always kept for
training). Modes:
  --fit train      fit on train queries (dense features excluded unless --dense_train), evaluate on dev
  --fit devcv      k-fold cross-validation over dev queries (out-of-fold dev F1), final model fit on all dev
  --fit both       fit on train + dev (CV over dev folds, train always in)
Test prediction from the final model; the query's own id is excluded; output = top 7.
"""

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

import pcr_data as D
from features import build, build_tfidf

RUNS = os.path.join(D.ROOT, "runs_p7")
LEX = ["uni_full", "uni_ctx", "bi_full", "bi_ctx", "tri_full", "tri_ctx"]
TFIDF = ["tf13_full", "tf13_ctx", "tf22_ctx", "tf33_full"]
FLAGS = ["date_viol", "year_viol", "year_gap", "is_query", "citator", "cite_text"]


def zr(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


def zc(S):
    return (S - S.mean(0, keepdims=True)) / (S.std(0, keepdims=True) + 1e-9)


def rr(S):
    r = np.argsort(np.argsort(-S, 1), 1)
    return 1.0 / (r + 1.0)


def dense_scores(run, split, qids, cids, view="doc"):
    z = np.load(os.path.join(RUNS, run, f"emb_{split}.npz"))
    ids = list(z["ids"])
    pos = {c: j for j, c in enumerate(ids)}
    V = z["V"]
    if view == "doc":
        if "QV" in z.files:     # query vectors stored separately (they differ from the pool vector with --query_view ctx)
            qp = {q: i for i, q in enumerate(z["qids"])}
            Qv = z["QV"][[qp[q] for q in qids]]
        else:
            Qv = V[[pos[q] for q in qids]]
        S = Qv @ V[[pos[c] for c in cids]].T
    else:   # maxsim: mean over query chunks of the best-matching candidate chunk
        off = np.concatenate([[0], np.cumsum(z["nc"])])
        C = z["C"]
        cand_idx = [pos[c] for c in cids]
        cc = np.concatenate([C[off[j]:off[j + 1]] for j in cand_idx])
        owner = np.concatenate([[k] * (off[j + 1] - off[j]) for k, j in enumerate(cand_idx)])
        S = np.zeros((len(qids), len(cids)), np.float32)
        for i, q in enumerate(qids):
            qc = C[off[pos[q]]:off[pos[q] + 1]]
            sim = qc @ cc.T                                     # (nq_chunks, all cand chunks)
            best = np.full((len(qc), len(cids)), -1.0, np.float32)
            np.maximum.at(best.T, owner, sim.T)
            S[i] = best.mean(0)
    return S.astype(np.float32)


def ce_features(run, split, qids, cids):
    """Cross-encoder scores (rerank_ce.py) on the stage-1 top-n: z within the reranked set (others -3), 1/(rank+1)
    within the set (others 0), and an in-set flag."""
    z = np.load(os.path.join(RUNS, run, f"ce_{split}.npz"))
    assert list(z["qids"]) == qids and list(z["cids"]) == cids
    S = z["S"]
    fin = np.isfinite(S)
    mu = np.where(fin, S, 0).sum(1, keepdims=True) / fin.sum(1, keepdims=True).clip(1)
    sd = np.sqrt(np.where(fin, (S - mu) ** 2, 0).sum(1, keepdims=True) / fin.sum(1, keepdims=True).clip(1)) + 1e-6
    zz = np.where(fin, (np.where(fin, S, 0) - mu) / sd, -3.0)
    r = np.argsort(np.argsort(-np.where(fin, S, -1e9), 1), 1)
    return {f"ce:{run}|z": zz.astype(np.float32), f"ce:{run}|rr": np.where(fin, 1.0 / (r + 1.0), 0).astype(np.float32)}


def split_matrices(split, dense, dense_view, use_tfidf=True, ce=()):
    qids, cids, F = build(split)
    M = {}
    pos = {c: j for j, c in enumerate(cids)}
    selfmask = np.zeros((len(qids), len(cids)), bool)
    for i, q in enumerate(qids):
        selfmask[i, pos[q]] = True
    srcs = {k: F[k] for k in LEX}
    if use_tfidf:
        srcs.update(build_tfidf(split))
    for run in dense:
        if not os.path.exists(os.path.join(RUNS, run, f"emb_{split}.npz")):
            raise FileNotFoundError(f"{run} emb_{split}")
        srcs[f"d:{run}"] = dense_scores(run, split, qids, cids, dense_view)
    for k, S in srcs.items():
        S = np.where(selfmask, np.nan, S)
        S = np.where(selfmask, np.nanmin(S, 1, keepdims=True), S)
        z = zr(S)
        M[f"{k}|z"] = z
        M[f"{k}|cz"] = zc(z)
        M[f"{k}|rr"] = rr(S)
    for k in FLAGS:
        M[k] = F[k] if k != "year_gap" else np.clip(F[k], -20, 60) / 20.0
    lexsum = sum(M[f"{k}|z"] for k in srcs if not k.startswith("d:"))
    for run in ce:
        M.update(ce_features(run, split, qids, cids))
    return qids, cids, M, lexsum, selfmask


def pairs(qrows, M, lexsum, cand, gold, qids, cids, selfmask, keep_pos):
    pos = {c: j for j, c in enumerate(cids)}
    names = sorted(M)
    X, y, qi = [], [], []
    for i in qrows:
        order = np.argsort(-np.where(selfmask[i], -np.inf, lexsum[i]))[:cand]
        sel = set(order.tolist())
        if keep_pos:
            sel |= {pos[g] for g in gold.get(qids[i], []) if g in pos}
        sel = sorted(sel)
        X.append(np.stack([M[n][i, sel] for n in names], 1))
        g = set(gold.get(qids[i], []))
        y.append(np.array([cids[j] in g for j in sel], np.float32))
        qi.append(np.array([(i, j) for j in sel]))
    return np.concatenate(X), np.concatenate(y), np.concatenate(qi), names


def predict_scores(model, qrows, M, lexsum, cand, selfmask, names):
    """-> {query row: (candidate columns, LR logits)} over the top `cand` lexical candidates."""
    out = {}
    for i in qrows:
        order = np.argsort(-np.where(selfmask[i], -np.inf, lexsum[i]))[:cand]
        order = np.array([j for j in order if not selfmask[i, j]])
        X = np.stack([M[n][i, order] for n in names], 1)
        out[i] = (order, model.decision_function(X))
    return out


def assign(scores, qids, cids, cap=None):
    """Top-7 per query; with cap, a global greedy pass (highest logit first) lets each candidate fill at most `cap`
    queries' lists, the remaining slots are filled in score order afterwards (so every list still has 7)."""
    if cap is None:
        return {qids[i]: [cids[j] for j in o[np.argsort(-s, kind="stable")][:D.K]] for i, (o, s) in scores.items()}
    trip = sorted(((s[k], i, j) for i, (o, s) in scores.items() for k, j in enumerate(o)), reverse=True)
    lists = {i: [] for i in scores}
    used = {}
    for s, i, j in trip:
        if len(lists[i]) < D.K and used.get(j, 0) < cap:
            lists[i].append(j)
            used[j] = used.get(j, 0) + 1
    for i, (o, s) in scores.items():
        for j in o[np.argsort(-s, kind="stable")]:
            if len(lists[i]) >= D.K:
                break
            if j not in lists[i]:
                lists[i].append(j)
    return {qids[i]: [cids[j] for j in v] for i, v in lists.items()}


def predict_rank(model, qrows, M, lexsum, cand, qids, cids, selfmask, names, cap=None):
    return assign(predict_scores(model, qrows, M, lexsum, cand, selfmask, names), qids, cids, cap)


def fit(X, y, C):
    m = LogisticRegression(C=C, max_iter=3000, class_weight=None)
    m.fit(X, y)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", nargs="*", default=[])
    ap.add_argument("--ce", nargs="*", default=[], help="cross-encoder runs (rerank_ce.py), dev/test only")
    ap.add_argument("--dense_view", default="doc", choices=["doc", "maxsim"])
    ap.add_argument("--fit", default="devcv", choices=["train", "devcv", "both", "stack"],
                    help="stack: stage 1 = lexical+metadata LR fit on train (no dense); stage 2 = dev-CV LR on "
                         "[stage-1 logit, dense features, is_query]")
    ap.add_argument("--drop", nargs="*", default=[], help="feature name substrings to exclude")
    ap.add_argument("--cand", type=int, default=150)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3, help="CV repetitions with different fold splits")
    ap.add_argument("--isq_shift", type=float, default=None,
                    help="added to the is_query weight for test (prior correction for the test pool's query share)")
    ap.add_argument("--cap", type=int, default=None, help="global capacity per candidate (assign())")
    ap.add_argument("--out", default=None, help="write test submission here")
    a = ap.parse_args()

    dv = split_matrices("dev", a.dense, a.dense_view, ce=a.ce)
    qd, cd, Md, Ld, SMd = dv
    _, gold_d, _ = D.load("dev")

    def filt(M):
        return {k: v for k, v in M.items() if not any(s in k for s in a.drop)}
    Md = filt(Md)
    res = {}
    base = None
    if a.fit == "stack":
        qt, ct, Mt, Lt, SMt = split_matrices("train", [], a.dense_view)
        Mt = filt(Mt)
        _, gold_t, _ = D.load("train")
        X, y, _, bnames = pairs(range(len(qt)), Mt, Lt, a.cand, gold_t, qt, ct, SMt, True)
        base = fit(X, y, a.C)
        res["stage1_dev"] = D.score(gold_d, predict_rank(base, range(len(qd)), Md, Ld, a.cand, qd, cd, SMd, bnames))

        def stage2(M):
            full = np.stack([M[n] for n in bnames], -1)
            logit = (full @ base.coef_[0] + base.intercept_[0]).astype(np.float32)
            out = {"base": logit, "is_query": M["is_query"]}
            out.update({k: v for k, v in M.items() if k.startswith(("d:", "ce:"))})
            return out
        Md = stage2(Md)
    if a.fit == "train":
        tr = split_matrices("train", [], a.dense_view) if not a.dense else None
        assert tr is not None, "--fit train uses no dense features"
        qt, ct, Mt, Lt, SMt = tr
        Mt = filt(Mt)
        _, gold_t, _ = D.load("train")
        X, y, _, names = pairs(range(len(qt)), Mt, Lt, a.cand, gold_t, qt, ct, SMt, True)
        model = fit(X, y, a.C)
        pred = predict_rank(model, range(len(qd)), Md, Ld, a.cand, qd, cd, SMd, names)
        res["dev"] = D.score(gold_d, pred)
        final = model
    else:
        extra = None
        if a.fit == "both":
            qt, ct, Mt, Lt, SMt = split_matrices("train", [], a.dense_view)
            _, gold_t, _ = D.load("train")
            extra = pairs(range(len(qt)), filt(Mt), Lt, a.cand, gold_t, qt, ct, SMt, True)
        f1s, cap_f1 = [], {}
        for seed in range(a.seeds):
            perm = np.random.RandomState(seed).permutation(len(qd))
            sc = {}
            for f in range(a.folds):
                te = perm[f::a.folds]
                trn = np.setdiff1d(perm, te)
                X, y, _, names = pairs(trn, Md, Ld, a.cand, gold_d, qd, cd, SMd, True)
                if extra is not None:
                    X, y = np.concatenate([X, extra[0]]), np.concatenate([y, extra[1]])
                model = fit(X, y, a.C)
                sc.update(predict_scores(model, te, Md, Ld, a.cand, SMd, names))
            oof = assign(sc, qd, cd, a.cap)
            if seed == 0:
                oof0 = oof
            f1s.append(D.score(gold_d, oof)["f1"])
            for cap in (1, 2, 3):
                cap_f1.setdefault(cap, []).append(D.score(gold_d, assign(sc, qd, cd, cap))["f1"])
        res["dev_cv"] = {"f1_mean": float(np.mean(f1s)), "f1_seeds": [round(x, 4) for x in f1s],
                         **{f"cap{c}": round(float(np.mean(v)), 4) for c, v in cap_f1.items()}}
        X, y, _, names = pairs(range(len(qd)), Md, Ld, a.cand, gold_d, qd, cd, SMd, True)
        if extra is not None:
            X, y = np.concatenate([X, extra[0]]), np.concatenate([y, extra[1]])
        final = fit(X, y, a.C)
        res["dev_insample"] = D.score(gold_d, predict_rank(final, range(len(qd)), Md, Ld, a.cand, qd, cd, SMd, names, a.cap))
    coef = dict(zip(names, np.round(final.coef_[0], 3).tolist()))
    print(json.dumps({"args": vars(a), **res}, indent=None))
    print("coef:", json.dumps(dict(sorted(coef.items(), key=lambda kv: -abs(kv[1])))))
    if a.out:
        qt_, ct_, Mt_, Lt_, SMt_ = split_matrices("test", a.dense, a.dense_view, ce=a.ce)
        Mt_ = filt(Mt_)
        if base is not None:
            Mt_ = stage2(Mt_)
        if a.isq_shift is not None and "is_query" in names:
            final.coef_[0][names.index("is_query")] += a.isq_shift
        pred = predict_rank(final, range(len(qt_)), Mt_, Lt_, a.cand, qt_, ct_, SMt_, names, a.cap)
        p = D.write_submission(pred, ct_, qt_, a.out)
        json.dump({"args": vars(a), **res, "coef": coef}, open(os.path.join(a.out, "ltr.json"), "w"), indent=1)
        if a.fit != "train":    # out-of-fold dev predictions (CV seed 0) for the official evaluator
            json.dump(oof0, open(os.path.join(a.out, "dev_oof_prediction.json"), "w"), indent=1)
            json.dump(cd, open(os.path.join(a.out, "dev_candidate_ids.json"), "w"))
        print("wrote", p)


if __name__ == "__main__":
    main()
