"""Multi-label decision rules and scores for LSI (official metric: sklearn macro-F1 over the 100 statute columns).

Decision rules on sigmoid probabilities P (n, L):
  global     one threshold for every label
  per-label  one threshold per label, each chosen to maximise that label's F1 (macro-F1 is a mean of per-label F1s,
             so per-label tuning maximises it directly)
  atleast1   every document gets at least one statute: if nothing passes, the label with the largest P / thr
The honest estimate of a tuned rule is cross-fitted: tune on one half of dev, score the other half, swap, repeat.
"""

import numpy as np
from sklearn.metrics import f1_score

GRID = np.round(np.arange(0.02, 0.951, 0.01), 2)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decide(P, thr, atleast1=True):
    pred = P > thr
    if atleast1:
        empty = ~pred.any(1)
        if empty.any():
            pred[empty, np.argmax(P[empty] / thr, 1)] = True
    return pred


def scores(Y, pred):
    Y = Y.astype(bool)
    return {"macro": f1_score(Y, pred, average="macro", zero_division=0),
            "micro": f1_score(Y, pred, average="micro", zero_division=0),
            "subset": float((Y == pred).all(1).mean())}


def per_label_f1(Y, pred):
    Y = Y.astype(bool)
    tp = (Y & pred).sum(0)
    return 2 * tp / np.maximum(1, Y.sum(0) + pred.sum(0))


def tune_per_label(Y, P, grid=GRID):
    """-> (L,) thresholds, each the grid value with the best F1 for its label (ties: the middle one)."""
    Y = Y.astype(bool)
    thr = np.empty(Y.shape[1])
    for j in range(Y.shape[1]):
        pr = P[:, j][:, None] > grid[None]                                   # (n, G)
        tp = (pr & Y[:, j][:, None]).sum(0)
        f = 2 * tp / np.maximum(1, Y[:, j].sum() + pr.sum(0))
        best = np.flatnonzero(f == f.max())
        thr[j] = grid[best[len(best) // 2]]
    return thr


def tune_global(Y, P, grid=GRID):
    f = [f1_score(Y, P > t, average="macro", zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f))])


def crossfit(Y, P, reps=5, seed=0, atleast1=True):
    """Mean macro-F1 of per-label thresholds tuned on one random half of the docs and applied to the other."""
    rs = np.random.RandomState(seed)
    out = []
    for _ in range(reps):
        perm = rs.permutation(len(Y))
        a, b = perm[: len(Y) // 2], perm[len(Y) // 2:]
        for s, t in ((a, b), (b, a)):
            thr = tune_per_label(Y[s], P[s])
            out.append(scores(Y[t], decide(P[t], thr, atleast1))["macro"])
    return float(np.mean(out))


def report(Y, P):
    """All rules on one set of dev probabilities -> dict (+ the per-label thresholds tuned on all of dev)."""
    thr_l = tune_per_label(Y, P)
    thr_g = tune_global(Y, P)
    L = P.shape[1]
    rec = {}
    for k, thr, a1 in (("p05", np.full(L, 0.5), False), ("glob", np.full(L, thr_g), True),
                       ("lab", thr_l, False), ("lab1", thr_l, True)):
        rec.update({f"{k}_{n}": round(v, 4) for n, v in scores(Y, decide(P, thr, a1)).items()})
    rec["glob_thr"] = thr_g
    rec["cf_lab1"] = round(crossfit(Y, P, atleast1=True), 4)
    rec["cf_lab"] = round(crossfit(Y, P, atleast1=False), 4)
    return rec, thr_l
