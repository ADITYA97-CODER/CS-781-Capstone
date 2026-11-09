"""Cross-validation score: pool the held-out predictions of the CV halves (train_rr.py --epoch_logp) and compute the
official metric (13-label macro-F1) per epoch over all held-out sentences.

  python cv_score.py --runs ../runs_p3/cv/lorma_mlp_la_f0 ../runs_p3/cv/lorma_mlp_la_f1
"""

import argparse
import glob
import json
import os
import re

import numpy as np

import rr_data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()
    gold = {d["id"]: (d["domain"], np.array([D.LAB2ID[l] for l in d["labels"]]))
            for dom in D.DOMAINS for part in ("train", "dev") for d in D.load_split(f"{dom}_{part}")}
    epochs = sorted({int(re.search(r"ep(\d+)", f).group(1)) for f in glob.glob(os.path.join(args.runs[0], "dev_logp_ep*.npz"))})
    rows = []
    for e in epochs:
        paths = [os.path.join(r, f"dev_logp_ep{e}.npz") for r in args.runs]
        if not all(os.path.exists(p) for p in paths):
            break
        pred = {}
        for p in paths:
            z = np.load(p)
            pred.update({k: z[k].astype(np.float32).argmax(-1) for k in z.files})
        ids = sorted(pred)
        y = np.concatenate([gold[k][1] for k in ids])
        yp = np.concatenate([pred[k] for k in ids])
        f1, p, r, per = D.macro_scores(y, yp)
        dom = {}
        for d in D.DOMAINS:
            sel = [k for k in ids if gold[k][0] == d]
            dom[d] = D.macro_scores(np.concatenate([gold[k][1] for k in sel]), np.concatenate([pred[k] for k in sel]))[0]
        rows.append({"epoch": e, "f1": f1, "p": p, "r": r, "f1_CL": dom["CL"], "f1_IT": dom["IT"],
                     "acc": float((y == yp).mean()), "n_docs": len(ids), "n_sents": len(y),
                     "per_label": {l: round(v[2], 4) for l, v in per.items()}})
    for x in rows:
        print(f"ep {x['epoch']:2d}  CV macro-F1 {x['f1']:.4f}  P {x['p']:.3f} R {x['r']:.3f}  CL {x['f1_CL']:.4f} "
              f"IT {x['f1_IT']:.4f}  acc {x['acc']:.3f}  ({x['n_docs']} docs, {x['n_sents']} sents)")
    best = max(rows, key=lambda x: x["f1"])
    print(f"best epoch {best['epoch']}: {best['f1']:.4f}; per label {best['per_label']}")
    json.dump(rows, open(os.path.join(os.path.dirname(args.runs[0].rstrip('/')), os.path.basename(args.runs[0].rstrip('/'))[:-3] + "_cv.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
