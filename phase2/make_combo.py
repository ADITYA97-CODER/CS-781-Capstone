"""Per-label combination of the BIO-head and span-head models (P2.6), validated and packaged for submission.

  python make_combo.py --bio_runs ../runs_p2/e_sftup_lorma_all_s0 ../runs_p2/e_sftup_lorma_all_s1 \
      --span_runs ../runs_p2/e_sftup_lorma_span_all_s0 ../runs_p2/e_sftup_lorma_span_all_s1 \
      --out ../outputs/phase2/combo_lorma_bio_span

BIO runs: token probabilities (test_logp.npz) averaged over seeds and decoded as usual. Span runs: span candidate
probabilities (test_cands.npz) averaged over seeds and decoded at --threshold. The span-head entities are used for
--span_labels (default: the labels where the span head won on both CV folds), the BIO entities for every other label;
a BIO entity that overlaps a chosen span-head entity is dropped. Same rule as the CV check (mean 0.5896, in-sample).
"""

import argparse
import json
import os
import zipfile

import numpy as np

import ner_data as D
import span_head as S
from make_submission import validate


def combine(bio, span, span_labels):
    out = {}
    for k in bio:
        sb = [x for x in span[k] if x[2] in span_labels]
        sa = [x for x in bio[k] if x[2] not in span_labels and not any(y[0] < x[1] and y[1] > x[0] for y in sb)]
        out[k] = sorted(sa + sb)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bio_runs", nargs="+", required=True)
    ap.add_argument("--span_runs", nargs="+", required=True)
    ap.add_argument("--span_labels", nargs="+", default=["RESP", "STAT", "PREC", "CASENO"])
    ap.add_argument("--threshold", type=float, default=0.5)
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
    crf = [os.path.join(r, "crf.npz") for r in args.bio_runs]
    if all(os.path.exists(c) for c in crf):
        z = [np.load(c) for c in crf]
        D.set_crf([np.mean([x[f"arr_{i}"] for x in z], 0) for i in range(3)])
        print(f"CRF scores averaged over {len(z)} runs")
    texts, _ = D.load_split("fold3")
    tok = D.tokenizer()
    logps = [np.load(os.path.join(r, "test_logp.npz")) for r in args.bio_runs]
    cands = [S.load_candidates(os.path.join(r, "test_cands.npz")) for r in args.span_runs]
    bio, span = {}, {}
    for k, text in texts.items():
        _, offs = D.encode_doc(tok, text)
        probs = np.mean([np.exp(lp[k]) for lp in logps], axis=0)
        bio[k] = D.decode_doc(text, offs, D.apply_bias(np.log(probs + 1e-12), bias))
        span[k] = S.decode_doc_spans(text, offs, S.average_candidates([c[k] for c in cands]), args.threshold)
    pred = combine(bio, span, set(args.span_labels))
    validate(pred, texts)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "prediction.json")
    json.dump(pred, open(path, "w"))
    for name, p in (("bio_only", bio), ("span_only", span)):
        json.dump(p, open(os.path.join(args.out, f"{name}_prediction.json"), "w"))
    with zipfile.ZipFile(os.path.join(args.out, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, arcname="prediction.json")
    json.dump(vars(args), open(os.path.join(args.out, "sources.json"), "w"), indent=1)
    counts = {l: sum(x[2] == l for v in pred.values() for x in v) for l in D.LABELS}
    print(f"{len(pred)} docs, {sum(counts.values())} entities: {counts}")
    print(f"agreement with BIO-only: {D.strict_scores(bio, pred)[0]:.3f}, with span-only: {D.strict_scores(span, pred)[0]:.3f}")


if __name__ == "__main__":
    main()
