"""Knowledge distillation, step 1: token logits of trained taggers (the teachers) on the training documents.

  python kd_teacher.py --runs ../runs_p2/e_sftup_lorma_all_s0 ../runs_p2/e_sftup_lorma_all_s1 \
      ../runs_p2/e_sftup_lorma_flert_all_s0 ../runs_p2/e_sftup_lorma_flert_all_s1 \
      --splits fold1 fold2 --out ../runs_p2/kd_teacher_bio4.npz --gpu 1

Each run is rebuilt from its config.json (BIO head, adapters, doc or FLERT windows) with tagger_final.pt, and
predicts every document with its own windows; window logits are merged per token as at test time
(`merge_window_scores`). Output npz: {doc_id: (n_teachers, n_tokens, n_tags) float16 logits}. The student
(train_ner.py --kd) turns them into soft targets mean_i softmax(z_i / T).
"""

import argparse
import json
import os

import numpy as np
import torch

import ner_data as D
from ner_model import Tagger, load_backbone
from peft import add_adapters
from train_ner import prepare


@torch.no_grad()
def run_logits(model, win, device, batch_size):
    model.eval()
    out = []
    for i in range(0, len(win), batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(win.ids[i:i + batch_size].to(device), win.valid[i:i + batch_size].to(device))
        out.extend(logits.float().cpu().numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["fold1", "fold2"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    tok = D.tokenizer()
    texts, gold, docs = prepare(args.splits, tok)
    n_tok = [len(d[1]) for d in docs]
    per_teacher = []
    for run in args.runs:
        cfg = json.load(open(os.path.join(run, "config.json")))
        assert cfg.get("head", "bio") == "bio", run
        backbone, _ = load_backbone(cfg["ckpt"])
        model = Tagger(backbone, len(D.TAGS), bidirectional=not cfg["causal"], dropout=cfg["dropout"])
        if cfg["peft"] != "none":
            add_adapters(model.backbone, cfg["peft"], cfg["rank"], cfg["alpha"])
        model.load_state_dict(torch.load(os.path.join(run, "tagger_final.pt"), map_location="cpu"))
        model = model.to(device)
        if cfg.get("windows", "doc") == "flert":
            bounds = [D.sentence_bounds(texts[d[0]], d[2]) for d in docs]
            win = D.FlertWindows(docs, bounds, cfg["header_len"], cfg["ctx"], cfg["unit_len"], pack=not cfg["no_pack"])
        else:
            win = D.DocWindows(docs, cfg["header_len"], cfg["content_len"], cfg["eval_stride"])
        merged = D.merge_window_scores(n_tok, win.meta, run_logits(model, win, device, args.batch_size))
        # sanity: the teacher's hard predictions against gold (training docs, so close to perfect)
        pred = {d[0]: D.decode_doc(texts[d[0]], d[2], torch.log_softmax(torch.from_numpy(m), -1).numpy())
                for d, m in zip(docs, merged)}
        print(f"{run}: {len(win)} windows, train-doc macro-F1 {D.strict_scores(gold, pred)[0]:.4f}", flush=True)
        per_teacher.append(merged)
        del model, backbone
        torch.cuda.empty_cache()

    out = {d[0]: np.stack([t[i] for t in per_teacher]).astype(np.float16) for i, d in enumerate(docs)}
    np.savez_compressed(args.out, **out)
    json.dump({"runs": args.runs, "splits": args.splits}, open(args.out.replace(".npz", ".json"), "w"), indent=1)
    print(f"wrote {args.out}: {len(out)} docs, {len(args.runs)} teachers", flush=True)


if __name__ == "__main__":
    main()
