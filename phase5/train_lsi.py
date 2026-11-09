"""Fine-tune a Phase-1 model (Arch E / F) for Legal Statute Identification (100 statutes, multi-label, macro-F1).

  # LoRMA+ on up to 8 x 1024-token chunks spread over the document, attention pooling over chunks
  python train_lsi.py --view spread --max_len 1024 --max_chunks 8 --doc_pool attn --peft lorma --lr 5e-4 --out ...

Encoder + chunk/document pooling: phase4/cjpe_model.DocClassifier (bidirectional Phase-1 blocks over packed chunks)
with a 100-way head. Loss: binary cross-entropy per statute (--loss bce) or asymmetric loss (--loss asl, Ridnik et al.
2021: focusing gamma_neg on negatives + probability margin clip, for the ~96% negative labels). Selection and
per-label thresholds: dev (10,181 docs); the selection score is the cross-fitted per-label-threshold macro-F1
(cf_lab1, lsi_metrics.crossfit). Test logits only at evaluations that improve it.
Outputs in --out: config.json, log.jsonl, dev_logits.npz / test_logits.npz (ids, logits of the best evaluation),
best.json, thresholds.npy, classifier.pt (--save), prediction.json + submission.zip.
"""

import argparse
import json
import math
import os
import random
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

import lsi_data as D
import lsi_metrics as M

sys.path.insert(0, os.path.join(D.ROOT, "phase4"))
sys.path.insert(0, os.path.join(D.ROOT, "phase2"))
from cjpe_model import DocClassifier, load_backbone  # noqa: E402
from lsi_model import LWANClassifier  # noqa: E402
from peft import add_adapters  # noqa: E402


def asl_loss(logits, y, gamma_neg=4.0, gamma_pos=0.0, clip=0.05):
    """Asymmetric loss (Ridnik et al. 2021), summed over labels -> (n_docs,)."""
    p = torch.sigmoid(logits)
    p_neg = (p - clip).clamp(min=0) if clip > 0 else p                 # shifted probability for negatives
    lp = F.logsigmoid(logits)
    ln = torch.log((1 - p_neg).clamp(min=1e-8))
    loss = y * (1 - p) ** gamma_pos * lp + (1 - y) * p_neg ** gamma_neg * ln
    return -loss.sum(-1)


@torch.no_grad()
def predict(model, chunks, doc_len, args, device):
    model.eval()
    order = sorted(range(len(chunks)), key=lambda d: -sum(len(c) for c, _ in chunks[d]))
    out = np.zeros((len(chunks), D.N_LABELS), dtype=np.float32)
    for mb in D.micro_batches(chunks, order, args.rows, args.seq):
        b = D.pack(mb, chunks, doc_len, args.rows, args.seq, args.n_pos, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(b)
        out[b["docs"]] = logits.float().cpu().numpy()
    model.train()
    return out


def write_submission(ids, pred, out):
    assert len(ids) == len(set(ids)) == len(pred)
    recs = [{"id": i, "pred": sorted(int(j) for j in np.flatnonzero(p))} for i, p in zip(ids, pred)]
    assert all(0 <= j < D.N_LABELS for r in recs for j in r["pred"])
    path = os.path.join(out, "prediction.json")
    json.dump(recs, open(path, "w"))
    with zipfile.ZipFile(os.path.join(out, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, "prediction.json")
    print(f"wrote {len(recs)} test predictions ({np.mean([len(r['pred']) for r in recs]):.2f} labels/doc) "
          f"-> {out}/submission.zip", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(D.ROOT, "runs/sft_archE_v2_upweight/ckpt_best.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--view", default="spread", choices=["last", "first", "headtail", "hier", "spread"])
    ap.add_argument("--max_len", type=int, default=1024, help="tokens per chunk")
    ap.add_argument("--max_chunks", type=int, default=8, help="hier / spread: chunks per document")
    ap.add_argument("--head_len", type=int, default=256, help="headtail: tokens from the start")
    ap.add_argument("--rows", type=int, default=4, help="packed rows per micro-batch")
    ap.add_argument("--seq", type=int, default=4096, help="tokens per packed row")
    ap.add_argument("--accum", type=int, default=2, help="micro-batches per optimizer step")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--evals_per_epoch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--doc_pool", default="attn", choices=["mean", "attn"])
    ap.add_argument("--n_pos", type=int, default=8, help="relative-position buckets of a chunk in its document")
    ap.add_argument("--mlp", action="store_true")
    ap.add_argument("--head", default="pool", choices=["pool", "lwan"],
                    help="lwan: + label-wise attention over all tokens (lsi_model.py)")
    ap.add_argument("--loss", default="bce", choices=["bce", "asl"])
    ap.add_argument("--asl_gamma_neg", type=float, default=4.0)
    ap.add_argument("--asl_clip", type=float, default=0.05)
    ap.add_argument("--peft", default="none", choices=["none", "lorma", "lora", "rslora", "bitfit", "ia3"])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--train_layers", type=int, default=None,
                    help="partial fine-tuning: train only the top k blocks (+ final norm, head), freeze the rest")
    ap.add_argument("--n_layers", type=int, default=None, help="student: keep only the first n_layers blocks")
    ap.add_argument("--kd", default=None, help="teacher logits npz (ids, logits) over the train docs")
    ap.add_argument("--kd_T", type=float, default=2.0)
    ap.add_argument("--kd_alpha", type=float, default=0.5, help="weight of the gold loss (1 - alpha on T^2 soft BCE)")
    ap.add_argument("--max_train_docs", type=int, default=None, help="smoke tests")
    ap.add_argument("--max_dev_docs", type=int, default=None, help="smoke tests")
    ap.add_argument("--no_test", action="store_true", help="smoke tests: no test predictions")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true", help="save the trainable weights of the best evaluation")
    ap.add_argument("--inference", action="store_true", help="no training: load <out>/classifier.pt and predict dev + test")
    ap.add_argument("--predict_train", action="store_true", help="also save logits of the train docs (KD teacher)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    def prep(name, limit=None):
        ids, Y, toks = D.load_tokens(name)
        if limit:
            sel = np.random.RandomState(0).permutation(len(ids))[:limit]
            ids, Y, toks = [ids[i] for i in sel], Y[sel], [toks[i] for i in sel]
        ch = [D.doc_chunks(t, args.view, args.max_len, args.max_chunks, args.head_len) for t in toks]
        return ids, Y, ch, [len(t) for t in toks]
    tr_ids, tr_Y, tr_ch, tr_len = prep("train", args.max_train_docs)
    dv_ids, dv_Y, dv_ch, dv_len = prep("dev", args.max_dev_docs)
    te_ids, _, te_ch, te_len = prep("test")
    soft = None
    if args.kd:
        z = np.load(args.kd)
        pos = {i: k for k, i in enumerate(z["ids"].astype(str))}
        soft = torch.sigmoid(torch.from_numpy(z["logits"][[pos[i] for i in tr_ids]].astype(np.float32)) / args.kd_T)
        print(f"KD teacher {args.kd}: T {args.kd_T}, alpha {args.kd_alpha}", flush=True)

    backbone, cfg = load_backbone(args.ckpt, max(4096, args.max_len))
    if args.n_layers:
        backbone.blocks = backbone.blocks[:args.n_layers]
        print(f"student: first {args.n_layers} of {cfg.n_layers} blocks", flush=True)
    model = (LWANClassifier if args.head == "lwan" else DocClassifier)(
        backbone, D.N_LABELS, args.dropout, args.doc_pool, args.n_pos, args.mlp)
    last = model.head[3] if args.mlp else model.head                  # start at the label priors
    prior = tr_Y.mean(0).clip(1e-4, 1 - 1e-4)
    with torch.no_grad():
        last.bias.copy_(torch.from_numpy(np.log(prior / (1 - prior))).float())
    if args.peft != "none":
        n = add_adapters(model.backbone, args.peft, args.rank, args.alpha)
        print(f"{args.peft}: {n} linear layers adapted", flush=True)
    elif args.train_layers is not None:
        nb = len(model.backbone.blocks)
        for name, p in model.backbone.named_parameters():
            top = name.startswith("blocks.") and int(name.split(".")[1]) >= nb - args.train_layers
            p.requires_grad = top or name.startswith("norm.")
        print(f"partial FT: top {args.train_layers} of {nb} blocks trainable", flush=True)
    model = model.to(device)
    if not args.no_compile:
        from torch import _dynamo
        _dynamo.config.cache_size_limit = 64
        _dynamo.config.automatic_dynamic_shapes = False
        model.set_compiled(torch.compile(model.encoder, dynamic=False))
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    decay, no_decay = [], []
    for name, p in model.encoder.named_parameters():
        if p.requires_grad:
            (no_decay if p.dim() < 2 or "norm" in name or "lambda" in name or "subln" in name else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "lr": args.lr, "weight_decay": args.weight_decay},
                             {"params": no_decay, "lr": args.lr, "weight_decay": 0.0},
                             {"params": head_params, "lr": args.head_lr, "weight_decay": 0.0}],
                            betas=(0.9, 0.98), eps=1e-6)
    base_lrs = [g["lr"] for g in opt.param_groups]

    mb_epoch = len(D.micro_batches(tr_ch, list(range(len(tr_ch))), args.rows, args.seq))
    steps_epoch = math.ceil(mb_epoch / args.accum)
    total = int(steps_epoch * args.epochs)
    warm = max(1, int(args.warmup_frac * total))
    eval_at = {int(round(steps_epoch * k / args.evals_per_epoch))
               for k in range(1, int(math.ceil(args.epochs * args.evals_per_epoch)) + 1)}
    eval_at = {min(s, total) for s in eval_at}
    tr_tok = sum(len(c) for ch in tr_ch for c, _ in ch)
    print(f"{args.view} max_len {args.max_len} x{args.max_chunks} | params {n_params/1e6:.2f}M trainable "
          f"{n_train/1e6:.2f}M | train {len(tr_ids)} docs, {tr_tok/1e6:.1f}M tokens/epoch, {mb_epoch} micro-batches of "
          f"{args.rows}x{args.seq} ({tr_tok / (mb_epoch * args.rows * args.seq):.0%} filled) | {steps_epoch} steps/epoch, "
          f"{total} total | loss {args.loss} | dev {len(dv_ids)} test {len(te_ids)}", flush=True)

    if args.inference:
        model.load_state_dict(torch.load(os.path.join(args.out, "classifier.pt"), map_location="cpu"), strict=False)
        dv, te = predict(model, dv_ch, dv_len, args, device), predict(model, te_ch, te_len, args, device)
        np.savez(os.path.join(args.out, "dev_logits.npz"), ids=np.array(dv_ids), logits=dv)
        np.savez(os.path.join(args.out, "test_logits.npz"), ids=np.array(te_ids), logits=te)
        _, thr = M.report(dv_Y, M.sigmoid(dv))
        np.save(os.path.join(args.out, "thresholds.npy"), thr)
        write_submission(te_ids, M.decide(M.sigmoid(te), thr, True), args.out)
        return

    log = open(os.path.join(args.out, "log.jsonl"), "a")
    best, step, t0, seen_tok, epoch = -1.0, 0, time.time(), 0, 0
    model.train()
    while step < total:
        epoch += 1
        mbs = D.micro_batches(tr_ch, torch.randperm(len(tr_ch)).tolist(), args.rows, args.seq)
        tot_loss, n_loss = 0.0, 0
        for i in range(0, len(mbs), args.accum):
            if step >= total:
                break
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * (step / warm if step < warm else max(0.0, (total - step) / max(1, total - warm)))
            group = mbs[i:i + args.accum]
            n_docs = sum(len(mb) for mb in group)
            for mb in group:
                b = D.pack(mb, tr_ch, tr_len, args.rows, args.seq, args.n_pos, device)
                y = torch.from_numpy(tr_Y[b["docs"]]).to(device).float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(b)
                logits = logits.float()
                if args.loss == "asl":
                    loss = asl_loss(logits, y, args.asl_gamma_neg, 0.0, args.asl_clip)
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none").sum(-1)
                if soft is not None:
                    pt = soft[b["docs"]].to(device)
                    kl = F.binary_cross_entropy_with_logits(logits / args.kd_T, pt, reduction="none").sum(-1)
                    loss = args.kd_alpha * loss + (1 - args.kd_alpha) * args.kd_T ** 2 * kl
                (loss.sum() / n_docs).backward()
                tot_loss += loss.sum().item()
                n_loss += len(mb)
                seen_tok += int((b["tok_chunk"] >= 0).sum())
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                el = time.time() - t0
                print(f"step {step}/{total} loss {tot_loss / n_loss:.4f} {seen_tok / el / 1e3:.1f}k tok/s "
                      f"{el / 60:.1f} min, eta {(total - step) * el / step / 60:.0f} min", flush=True)
            if step in eval_at:
                te0 = time.time()
                dv = predict(model, dv_ch, dv_len, args, device)
                rec = {"step": step, "epoch": round(step / steps_epoch, 2),
                       "train_loss": round(tot_loss / max(1, n_loss), 5),
                       "time_min": round((time.time() - t0) / 60, 1),
                       "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 1)}
                r, thr = M.report(dv_Y, M.sigmoid(dv))
                rec.update(r)
                if rec["cf_lab1"] > best:
                    best = rec["cf_lab1"]
                    rec["best"] = True
                    np.savez(os.path.join(args.out, "dev_logits.npz"), ids=np.array(dv_ids), logits=dv)
                    np.save(os.path.join(args.out, "thresholds.npy"), thr)
                    json.dump(rec, open(os.path.join(args.out, "best.json"), "w"), indent=1)
                    if not args.no_test:
                        te = predict(model, te_ch, te_len, args, device)
                        np.savez(os.path.join(args.out, "test_logits.npz"), ids=np.array(te_ids), logits=te)
                        write_submission(te_ids, M.decide(M.sigmoid(te), thr, True), args.out)
                    if args.save:
                        keep = {n for n, p in model.named_parameters() if p.requires_grad}
                        torch.save({k: v for k, v in model.state_dict().items()
                                    if k in keep or args.peft == "none"},
                                   os.path.join(args.out, "classifier.pt"))
                rec["eval_min"] = round((time.time() - te0) / 60, 1)
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
    if args.predict_train:
        if args.save:
            model.load_state_dict(torch.load(os.path.join(args.out, "classifier.pt")), strict=False)
        tr = predict(model, tr_ch, tr_len, args, device)
        np.savez(os.path.join(args.out, "train_logits.npz"), ids=np.array(tr_ids), logits=tr)
    print(f"done in {(time.time() - t0) / 60:.0f} min; best dev macro-F1 (cross-fitted per-label thr) {best:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
