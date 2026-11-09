"""Generate summaries with a trained Phase-6 run, score them on dev with the official metrics, write test submissions.

  python gen_summ.py --run runs_p6/e_full_ht4k --split dev --beams 4 --no_repeat 4 --tag b4
  python gen_summ.py --run runs_p6/e_full_ht4k --split test --beams 4 --no_repeat 4 --tag b4

Outputs in --run/gen_<split>_<tag>/: prediction.json (list format), args.json, scores.json (dev: BERTScore F1 with
the evaluator's exact call, ROUGE-1 / ROUGE-L F1, mean words); test also submission.zip (prediction.json inside).
Length tag (if the run was trained with one): predicted words = LengthModel(document words) * --len_scale,
rounded to 50 like in training.
"""

import argparse
import json
import os
import time
import zipfile

import numpy as np
import torch

import summ_data as D
from summ_model import generate
from train_summ import load_trained


def score(preds, refs):
    """Official evaluate_phase6.py metrics. preds / refs: dicts id -> text (same ids)."""
    import importlib.util
    path = os.path.join(D.ROOT, "official_eval", "Evaluation Script", "evaluate_phase6.py")
    spec = importlib.util.spec_from_file_location("ev6", path)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    import bert_score
    keys = list(refs)
    p = [preds[k].strip() for k in keys]
    r = [refs[k].strip() for k in keys]
    sc = ev.rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    tok = ev.tokenizers.DefaultTokenizer(use_stemmer=True)
    r1, rl = [], []
    for a, b in zip(p, r):
        r1.append(sc.score(b, a)["rouge1"].fmeasure)
        pt, rt = tok.tokenize(a), tok.tokenize(b)
        z = ev.lcs(pt, rt) if pt and rt else 0
        pp, rr = (z / len(pt), z / len(rt)) if pt and rt else (0, 0)
        rl.append(0.0 if pp + rr == 0 else 2 * pp * rr / (pp + rr))
    _, _, f = bert_score.score(p, r, lang="en", batch_size=8, verbose=False, device="cuda")
    return {"bertscore_f1": float(f.mean()), "rouge1": float(np.mean(r1)), "rougeL": float(np.mean(rl)),
            "words": float(np.mean([len(x.split()) for x in p])), "n": len(keys),
            "per_doc": {k: [float(a), float(b), float(c)] for k, a, b, c in zip(keys, f.tolist(), r1, rl)}}


def write_submission(pred, out_dir):
    recs = [{"id": k, "summary": v} for k, v in pred.items()]
    assert all(r["summary"].strip() for r in recs) and len({r["id"] for r in recs}) == len(recs)
    path = os.path.join(out_dir, "prediction.json")
    json.dump(recs, open(path, "w"), indent=1, ensure_ascii=False)
    with zipfile.ZipFile(os.path.join(out_dir, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, "prediction.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--tag", default="default")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_docs", type=int, default=None)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--rep_pen", type=float, default=1.0)
    ap.add_argument("--no_repeat", type=int, default=4)
    ap.add_argument("--len_pen", type=float, default=1.0)
    ap.add_argument("--min_new", type=int, default=200)
    ap.add_argument("--max_new", type=int, default=1024)
    ap.add_argument("--len_scale", type=float, default=1.0)
    ap.add_argument("--fixed_len", type=int, default=None, help="use this length tag for every document")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    bb, ra = load_trained(args.run, device)
    out_dir = os.path.join(args.run, f"gen_{args.split}_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, "args.json"), "w"), indent=1)

    _, _, _, tr_sw, tr_dw = D.load_tokens("train")
    lm = D.LengthModel(tr_dw, tr_sw)
    ids, docs, _, _, dwords = D.load_tokens(args.split)
    if args.max_docs:
        ids, docs, dwords = ids[:args.max_docs], docs[:args.max_docs], dwords[:args.max_docs]
    prompts = []
    for d, w in zip(docs, dwords):
        n = None
        if ra["len_tag"]:
            n = D.length_bucket(args.fixed_len or lm(w, args.len_scale))
        v = D.doc_view(d, ra["view"], ra["doc_len"], ra["head_len"])
        prompts.append(torch.from_numpy(np.concatenate([v, D.prompt_tokens(n)])))
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    tok = D.tokenizer()
    pred, t0 = {}, time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(order), args.batch):
            idx = order[s:s + args.batch]
            outs = generate(bb, [prompts[i] for i in idx], args, device)
            for i, o in zip(idx, outs):
                text = tok.decode(o).strip()
                pred[ids[i]] = text or "The appeal was dismissed."
            print(f"{s + len(idx)}/{len(order)} docs, {(time.time() - t0) / 60:.1f} min", flush=True)
    pred = {k: pred[k] for k in ids}
    gen_min = (time.time() - t0) / 60
    if args.split == "dev":
        json.dump([{"id": k, "summary": v} for k, v in pred.items()], open(os.path.join(out_dir, "prediction.json"), "w"),
                  indent=1, ensure_ascii=False)
        refs = D.references("dev")
        s = score(pred, {k: refs[k] for k in ids})
        s["gen_min"] = round(gen_min, 1)
        json.dump(s, open(os.path.join(out_dir, "scores.json"), "w"), indent=1)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k != "per_doc"}),
              flush=True)
    else:
        write_submission(pred, out_dir)
        print(f"wrote {len(pred)} test summaries (mean {np.mean([len(v.split()) for v in pred.values()]):.0f} words) "
              f"-> {out_dir}/submission.zip", flush=True)


if __name__ == "__main__":
    main()
