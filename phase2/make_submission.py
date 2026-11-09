"""Average the test-set token log-probs of several runs, decode, validate and package submission.zip.

  python make_submission.py --runs ../runs_p2/e_sftup_full_all_s0 ../runs_p2/e_sftup_full_all_s1 \
      --out ../outputs/phase2/e_sftup_full_ens2

Each run directory must hold test_logp.npz (written by train_ner.py --predict fold3). Probabilities are
averaged across runs, then decoded with the same Viterbi + offset trims (learned from fold1 + fold2).
Writes <out>/prediction.json and <out>/submission.zip (prediction.json at the zip root).
"""

import argparse
import json
import os
import zipfile

import numpy as np

import ner_data as D


def validate(pred, texts):
    """The checks of the official evaluator plus offset range checks; raises on any problem."""
    assert set(pred) == set(texts), f"id mismatch: missing {len(set(texts) - set(pred))}, extra {len(set(pred) - set(texts))}"
    for k, ents in pred.items():
        seen = set()
        for x in ents:
            assert isinstance(x, list) and len(x) == 3, (k, x)
            s, e, l = x
            assert type(s) is int and type(e) is int and 0 <= s < e <= len(texts[k]), (k, x)
            assert l in D.LABELS, (k, x)
            assert (s, e, l) not in seen, ("duplicate", k, x)
            seen.add((s, e, l))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bias", default=None, help="per-label BIO score bias json from tune_bias.py")
    args = ap.parse_args()

    tr_texts, tr_gold = {}, {}
    for name in ("fold1", "fold2"):
        t, g = D.load_split(name)
        tr_texts.update(t)
        tr_gold.update(g)
    D.learn_trims(tr_texts, tr_gold)

    bias = json.load(open(args.bias))["bias"] if args.bias else None
    crf = [os.path.join(r, "crf.npz") for r in args.runs]
    if all(os.path.exists(c) for c in crf):
        z = [np.load(c) for c in crf]
        D.set_crf([np.mean([x[f"arr_{i}"] for x in z], 0) for i in range(3)])
        print(f"CRF scores averaged over {len(z)} runs")
    texts, _ = D.load_split("fold3")
    tok = D.tokenizer()
    logps = [np.load(os.path.join(r, "test_logp.npz")) for r in args.runs]
    pred = {}
    for k, text in texts.items():
        _, offs = D.encode_doc(tok, text)
        probs = np.mean([np.exp(lp[k]) for lp in logps], axis=0)
        assert probs.shape[0] == len(offs), k
        pred[k] = D.decode_doc(text, offs, D.apply_bias(np.log(probs + 1e-12), bias))
    validate(pred, texts)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "prediction.json")
    json.dump(pred, open(path, "w"))
    with zipfile.ZipFile(os.path.join(args.out, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, arcname="prediction.json")
    json.dump({"runs": args.runs, "bias": args.bias}, open(os.path.join(args.out, "sources.json"), "w"), indent=1)
    counts = {l: sum(x[2] == l for v in pred.values() for x in v) for l in D.LABELS}
    print(f"{len(pred)} docs, {sum(counts.values())} entities: {counts}")


if __name__ == "__main__":
    main()
