"""Average the sigmoid probabilities of several runs (dev_logits.npz / test_logits.npz of each, optional weights), tune
per-label thresholds on dev, report all decision rules + the cross-fitted dev macro-F1, write prediction.json +
submission.zip.
  python make_ensemble.py --runs ../runs_p5/a ../runs_p5/b --weights 2 1 --out ../outputs/phase5/ens_ab"""
import argparse
import json
import os

import numpy as np

import lsi_data as D
import lsi_metrics as M
from train_lsi import write_submission


def probs(run, split, ids):
    z = np.load(os.path.join(run, f"{split}_logits.npz"))
    pos = {i: k for k, i in enumerate(z["ids"].astype(str))}
    return M.sigmoid(z["logits"][[pos[i] for i in ids]].astype(np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    w = np.array(a.weights or [1.0] * len(a.runs))
    w = w / w.sum()
    dv_ids, dv_Y, _ = D.load_tokens("dev")
    te_ids, _, _ = D.load_tokens("test")
    dv = sum(wi * probs(r, "dev", dv_ids) for wi, r in zip(w, a.runs))
    te = sum(wi * probs(r, "test", te_ids) for wi, r in zip(w, a.runs))
    for r in a.runs:
        print(f"{r}: cf_lab1 {M.crossfit(dv_Y, probs(r, 'dev', dv_ids)):.4f}")
    rec, thr = M.report(dv_Y, dv)
    rec.update({"runs": a.runs, "weights": w.round(4).tolist()})
    print(json.dumps(rec))
    os.makedirs(a.out, exist_ok=True)
    json.dump(rec, open(os.path.join(a.out, "ensemble.json"), "w"), indent=1)
    np.save(os.path.join(a.out, "thresholds.npy"), thr)
    np.savez(os.path.join(a.out, "dev_probs.npz"), ids=np.array(dv_ids), probs=dv.astype(np.float32))
    np.savez(os.path.join(a.out, "test_probs.npz"), ids=np.array(te_ids), probs=te.astype(np.float32))
    write_submission(te_ids, M.decide(te, thr, True), a.out)


if __name__ == "__main__":
    main()
