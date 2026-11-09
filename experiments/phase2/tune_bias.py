"""Tune a per-label BIO score bias on cross-validation dev log-probs (decoder calibration, 11 Oct).

  python tune_bias.py --cv ../runs_p2/e_sftup_lorma_cvlp_cv1 ../runs_p2/e_sftup_lorma_cvlp_cv2 --out ../runs_p2/bias_lorma.json

Each CV run holds dev_logp.npz (best epoch) for its dev fold; trims are learned on the run's training fold, as in
train_ner.py. For a bias b added to the B-/I- columns of one label, the label's F1 changes almost only through that
label, so one decode per grid value (the same b for every label) gives every label's F1 at that b; each label then
takes its best b. Honest estimate: choose on one fold, score on the other (both directions). Final bias: chosen on
both folds pooled.
"""

import argparse
import json
import os

import numpy as np

import ner_data as D

GRID = [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]


def fold_data(run, tok):
    cfg = json.load(open(os.path.join(run, "config.json")))
    texts, gold = D.load_split(cfg["dev"])
    lp = np.load(os.path.join(run, "dev_logp.npz"))
    offs = {k: D.encode_doc(tok, t)[1] for k, t in texts.items()}
    return cfg, texts, gold, {k: lp[k].astype(np.float32) for k in texts}, offs


def decode(fold, bias):
    cfg, texts, gold, lp, offs = fold
    D.learn_trims(*D.load_split(cfg["train"][0]))
    return {k: D.decode_doc(texts[k], offs[k], D.apply_bias(lp[k], bias)) for k in texts}


def label_stats(gold, pred):
    return D.strict_scores(gold, pred)[3]           # {label: (p, r, f1, tp, fp, fn)}


def f1_from(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", nargs=2, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = D.tokenizer()
    folds = [fold_data(r, tok) for r in args.cv]
    # per fold, per grid value: per-label (tp, fp, fn)
    stats = [{b: label_stats(f[2], decode(f, {l: b for l in D.LABELS})) for b in GRID} for f in folds]

    def choose(which):
        out = {}
        for l in D.LABELS:
            tot = {b: [sum(stats[i][b][l][j] for i in which) for j in (3, 4, 5)] for b in GRID}
            out[l] = max(GRID, key=lambda b: (f1_from(*tot[b]), -abs(b)))
        return out

    def macro(i, bias):
        return D.strict_scores(folds[i][2], decode(folds[i], bias))[0]

    res = {"grid": GRID}
    base = [macro(i, None) for i in range(2)]
    cross = []
    for i in range(2):
        b = choose([1 - i])                           # chosen on the other fold
        cross.append(macro(i, b))
        res[f"chosen_on_fold_of_{os.path.basename(args.cv[1 - i])}"] = b
    final = choose([0, 1])
    insample = [macro(i, final) for i in range(2)]
    res.update({"bias": final, "base_f1": base, "cross_fold_f1": cross, "in_sample_f1": insample})
    print(json.dumps(res, indent=1))
    print(f"macro-F1 base {np.mean(base):.4f} | cross-fold tuned {np.mean(cross):.4f} | in-sample {np.mean(insample):.4f}")
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
