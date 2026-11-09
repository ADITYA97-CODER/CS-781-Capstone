"""Final Phase 3 submission (16 Oct): mean probability of several test predictions + first-order Viterbi smoothing.

  python make_final.py --members ../runs_p3/all/laya_nm_s0 ../runs_p3/cv/lorma_win2048:17 --lam 0.15 --out ../outputs/phase3/x

A member is either a run directory with test_logp.npz (e.g. an all-data run) or <cv prefix>:<epoch> (the two CV halves'
test_logp_ep<epoch>.npz, averaged). --lam 0 = plain argmax. The transition matrix comes from the gold label sequences of
all 90 labelled docs. --lam_from <OOF specs> picks lam on the pooled out-of-fold predictions of those CV configs instead.
"""

import argparse
import json
import os

import numpy as np

import rr_data as D
from stage2 import f1, load_feats, mean_logp, trans_matrix, viterbi
from train_rr import TEST, write_submission


def member_logp(m):
    if os.path.isdir(m):
        z = np.load(os.path.join(m, "test_logp.npz"))
        return {k: z[k].astype(np.float32) for k in z.files}
    return {k: v[0] for k, v in load_feats([m], "test").items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--lam_from", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    gold = [np.array([D.LAB2ID[l] for l in d["labels"]])
            for dom in D.DOMAINS for p in ("train", "dev") for d in D.load_split(f"{dom}_{p}")]
    T, S = trans_matrix(gold)
    lam = args.lam
    if args.lam_from:
        feats = load_feats(args.lam_from)
        gid = {d["id"]: np.array([D.LAB2ID[l] for l in d["labels"]])
               for dom in D.DOMAINS for p in ("train", "dev") for d in D.load_split(f"{dom}_{p}")}
        ids = sorted(gid)
        lps = [mean_logp(feats[k]) for k in ids]
        ys = [gid[k] for k in ids]
        grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
        sc = [f1(ys, [viterbi(lp, T, S, l) for lp in lps]) for l in grid]
        lam = grid[int(np.argmax(sc))]
        print("OOF lam grid", dict(zip(grid, np.round(sc, 4))), "->", lam)
    lam = lam or 0.0
    os.makedirs(args.out, exist_ok=True)
    te_docs = [d for n in TEST for d in D.load_split(n)]
    zs = [member_logp(m) for m in args.members]
    w = np.array(args.weights or [1.0] * len(zs))
    w = w / w.sum()
    pred, agree = {}, []
    for d in te_docs:
        probs = [np.exp(z[d["id"]]) for z in zs]
        lp = np.log(sum(wi * p for wi, p in zip(w, probs)) + 1e-9)
        y = viterbi(lp, T, S, lam) if lam > 0 else lp.argmax(-1)
        pred[d["id"]] = [D.LABELS[k] for k in y]
        agree.append(np.array([p.argmax(-1) == y for p in probs]))
    agree = np.concatenate(agree, 1).mean(1)
    write_submission(pred, te_docs, args.out)
    info = {"members": args.members, "weights": w.round(3).tolist(), "lam": lam,
            "agreement_with_final": [round(float(a), 4) for a in agree],
            "label_counts": {l: sum(x.count(l) for x in pred.values()) for l in D.LABELS}}
    json.dump(info, open(os.path.join(args.out, "ensemble.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
