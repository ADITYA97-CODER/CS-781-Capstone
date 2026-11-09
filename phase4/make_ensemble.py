"""Average the class log-probs of several runs (dev_logits.npz / test_logits.npz of each), tune the margin threshold on
dev_clean, report tuned + 2-fold cross-fitted dev macro-F1, write prediction.json + submission.zip.
  python make_ensemble.py --runs ../runs_p4/a ../runs_p4/b --out ../outputs/phase4/ens_ab"""
import argparse
import json
import os

import numpy as np

import cjpe_data as D
from train_cjpe import write_submission


def f1s(y, m, ts):
    p = m[None, :] > ts[:, None]
    yy = y[None, :] == 1
    tp, fp = (p & yy).sum(1), (p & ~yy).sum(1)
    fn, tn = (~p & yy).sum(1), (~p & ~yy).sum(1)
    return (tp / np.maximum(1, tp + (fp + fn) / 2) + tn / np.maximum(1, tn + (fp + fn) / 2)) / 2


def tune(y, m):
    s = np.sort(m)
    c = (s[1:] + s[:-1]) / 2
    lo, hi = np.quantile(s, [0.05, 0.95])
    c = c[(c >= lo) & (c <= hi)]
    return float(c[np.argmax(f1s(y, m, c))])


def logp(path, ids):
    z = np.load(path)
    lg = np.stack([z[i] for i in ids]).astype(np.float64)
    return lg - np.logaddexp(lg[:, :1], lg[:, 1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    w = np.array(a.weights or [1.0] * len(a.runs))
    w = w / w.sum()
    dv_ids, y, _ = D.load_tokens("dev_clean")
    te_ids = [r["id"] for r in D.load_split("test")]
    dv = sum(wi * logp(os.path.join(r, "dev_logits.npz"), dv_ids) for wi, r in zip(w, a.runs))
    te = sum(wi * logp(os.path.join(r, "test_logits.npz"), te_ids) for wi, r in zip(w, a.runs))
    m, mt = dv[:, 1] - dv[:, 0], te[:, 1] - te[:, 0]
    thr = tune(y, m)
    idx = np.random.RandomState(0).permutation(len(y))
    A, B = idx[:len(y) // 2], idx[len(y) // 2:]
    cf = (f1s(y[B], m[B], np.array([tune(y[A], m[A])]))[0] + f1s(y[A], m[A], np.array([tune(y[B], m[B])]))[0]) / 2
    res = {"runs": a.runs, "weights": w.tolist(), "thr": thr, "dev_tuned_f1": float(f1s(y, m, np.array([thr]))[0]),
           "dev_crossfit_f1": float(cf), "dev_argmax_f1": float(f1s(y, m, np.array([0.0]))[0])}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "ensemble.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}))
    write_submission(te_ids, (mt > thr).astype(int), a.out)


if __name__ == "__main__":
    main()
