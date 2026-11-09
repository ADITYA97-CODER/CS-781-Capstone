"""Average the test-set sentence probabilities of several runs (train_rr.py --predict -> test_logp.npz), take the
argmax, validate and package submission.zip (as phase2/make_submission.py).

  python make_ensemble.py --runs ../runs_p3/all/laya_nm_s0 ../runs_p3/all/laya_nm_s1 --out ../outputs/phase3/laya_nm_ens2
"""

import argparse
import json
import os

import numpy as np

import rr_data as D
from train_rr import TEST, write_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    te_docs = [d for n in TEST for d in D.load_split(n)]
    zs = [np.load(os.path.join(r, "test_logp.npz")) for r in args.runs]
    pred, agree = {}, []
    for d in te_docs:
        probs = [np.exp(z[d["id"]].astype(np.float32)) for z in zs]
        mean = np.mean(probs, 0)
        pred[d["id"]] = [D.LABELS[k] for k in mean.argmax(-1)]
        agree.append(np.array([p.argmax(-1) == mean.argmax(-1) for p in probs]))      # (runs, n_sents)
    agree = np.concatenate(agree, 1).mean(1)
    write_submission(pred, te_docs, args.out)
    info = {"runs": args.runs, "agreement_with_ensemble": [round(float(a), 4) for a in agree],
            "label_counts": {l: sum(x.count(l) for x in pred.values()) for l in D.LABELS}}
    json.dump(info, open(os.path.join(args.out, "ensemble.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
