"""Cross-system MBR: for every document pick the candidate summary with the highest mean BERTScore F1 (the official
scorer's model and truncation) against the other candidates. Candidates = prediction.json files of several systems
(or several samples of one system).

  python mbr_select.py --out outputs/phase6/mbr_x --split test --preds runs_p6/a/gen_test_b4/prediction.json ...

Dev split: also scores the selection (and every input) with the official metrics. Test: writes submission.zip.
--weights: per-input prior weights (the utility of candidate i is sum_j w_j F1(i, j) / sum_j w_j, j != i).
"""

import argparse
import json
import os

import numpy as np

import summ_data as D
from gen_summ import score, write_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import bert_score
    os.makedirs(args.out, exist_ok=True)
    cands = [{r["id"]: r["summary"] for r in json.load(open(p))} for p in args.preds]
    ids = list(cands[0])
    assert all(set(c) == set(ids) for c in cands)
    n = len(cands)
    w = np.array(args.weights or [1.0] * n)
    a, b, keys = [], [], []
    for k in ids:
        for i in range(n):
            for j in range(n):
                if i != j:
                    a.append(cands[i][k])
                    b.append(cands[j][k])
                    keys.append((k, i, j))
    scorer = bert_score.BERTScorer(lang="en", device="cuda")
    _, _, f = scorer.score(a, b, batch_size=32)
    util = {k: np.zeros(n) for k in ids}
    for (k, i, j), x in zip(keys, f.tolist()):
        util[k][i] += w[j] * x
    pick, choice = {}, []
    for k in ids:
        u = util[k] / (w.sum() - w)
        c = int(u.argmax())
        choice.append(c)
        pick[k] = cands[c][k]
    rep = {"inputs": args.preds, "weights": w.tolist(), "picks": np.bincount(choice, minlength=n).tolist()}
    if args.split == "dev":
        refs = D.references("dev")
        refs = {k: refs[k] for k in ids}
        rep["mbr"] = {x: v for x, v in score(pick, refs).items() if x != "per_doc"}
        rep["inputs_scores"] = [{x: v for x, v in score(c, refs).items() if x != "per_doc"} for c in cands]
        json.dump([{"id": k, "summary": v} for k, v in pick.items()], open(os.path.join(args.out, "prediction.json"), "w"),
                  indent=1, ensure_ascii=False)
    else:
        write_submission(pick, args.out)
    json.dump(rep, open(os.path.join(args.out, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
