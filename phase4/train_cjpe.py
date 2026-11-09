"""Fine-tune a Phase-1 model (Arch E / F) for Court Judgment Prediction (binary, macro-F1) and predict the test set.

  # LoRMA+ on the last 2048 tokens of every judgment
  python train_cjpe.py --view last --max_len 2048 --peft lorma --lr 5e-4 --out runs_p4/e_lorma_last2k --gpu 0
  # hierarchical: last 4 x 1024-token chunks, attention pooling over chunks
  python train_cjpe.py --view hier --max_len 1024 --max_chunks 4 --doc_pool attn --peft lorma --lr 5e-4 --out ...

Training data: the 31,677 unique train cases. Selection and the decision threshold: dev_clean (994 cases, none in
test). Test predictions are made only at evaluations that improve dev. Outputs in --out: config.json, log.jsonl,
dev_logits.npz / test_logits.npz (best evaluation), classifier.pt (--save), prediction.json + submission.zip.
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
from sklearn.metrics import f1_score, precision_score, recall_score

import cjpe_data as D
from cjpe_model import DocClassifier, load_backbone

sys.path.insert(0, os.path.join(D.ROOT, "phase2"))
from peft import add_adapters  # noqa: E402


def scores(y, margin, thr=0.0):
    """margin = logit1 - logit0; predict 1 where margin > thr."""
    p = (margin > thr).astype(int)
    return {"f1": f1_score(y, p, average="macro", labels=[0, 1], zero_division=0),
            "p": precision_score(y, p, zero_division=0), "r": recall_score(y, p, zero_division=0),
            "pos_rate": float(p.mean())}


def tune_threshold(y, margin):
    """Margin threshold with the best macro-F1 (candidates: midpoints of the sorted margins, restricted to the
    central 90% so that a single extreme document cannot set it)."""
    m = np.sort(margin)
    lo, hi = np.quantile(m, [0.05, 0.95])
    cand = (m[1:] + m[:-1]) / 2
    cand = cand[(cand >= lo) & (cand <= hi)]
    best = max(cand, key=lambda t: f1_score(y, (margin > t).astype(int), average="macro"))
    return float(best)


@torch.no_grad()
def predict(model, chunks, doc_len, args, device):
    model.eval()
    order = sorted(range(len(chunks)), key=lambda d: -sum(len(c) for c, _ in chunks[d]))
    out = np.zeros((len(chunks), 2), dtype=np.float32)
    for mb in D.micro_batches(chunks, order, args.rows, args.seq):
        b = D.pack(mb, chunks, doc_len, args.rows, args.seq, args.n_pos, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(b)
        out[b["docs"]] = logits.float().cpu().numpy()
    model.train()
    return out


def write_submission(ids, pred, out):
    assert len(ids) == len(set(ids)) == len(pred)
    recs = [{"id": i, "pred": int(p)} for i, p in zip(ids, pred)]
    assert all(r["pred"] in (0, 1) for r in recs)
    path = os.path.join(out, "prediction.json")
    json.dump(recs, open(path, "w"), indent=1)
    with zipfile.ZipFile(os.path.join(out, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, "prediction.json")
    print(f"wrote {len(recs)} test predictions ({np.mean(pred):.3f} positive) -> {out}/submission.zip", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(D.ROOT, "runs/sft_archE_v2_upweight/ckpt_best.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--view", default="last", choices=["last", "first", "headtail", "hier", "spread"])
    ap.add_argument("--max_len", type=int, default=2048, help="tokens per chunk")
    ap.add_argument("--max_chunks", type=int, default=1, help="hier / spread: chunks per document")
    ap.add_argument("--head_len", type=int, default=256, help="headtail: tokens from the start")
    ap.add_argument("--rows", type=int, default=8, help="packed rows per micro-batch")
    ap.add_argument("--seq", type=int, default=None, help="tokens per packed row (default max_len)")
    ap.add_argument("--accum", type=int, default=2, help="micro-batches per optimizer step")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--evals_per_epoch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--doc_pool", default="mean", choices=["mean", "attn"])
    ap.add_argument("--n_pos", type=int, default=8, help="relative-position buckets of a chunk in its document")
    ap.add_argument("--mlp", action="store_true")
    ap.add_argument("--class_weight", default="none", choices=["none", "balanced"])
    ap.add_argument("--peft", default="none", choices=["none", "lorma", "lora", "rslora", "bitfit", "ia3"])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--train_layers", type=int, default=None,
                    help="partial fine-tuning: train only the top k blocks (+ final norm, head), freeze the rest")
    ap.add_argument("--n_layers", type=int, default=None, help="student: keep only the first n_layers blocks")
    ap.add_argument("--kd", default=None, help="teacher logits npz (id -> 2 logits) over the train docs")
    ap.add_argument("--kd_T", type=float, default=2.0)
    ap.add_argument("--kd_alpha", type=float, default=0.5, help="weight of the gold CE (1 - alpha on T^2 KL)")
    ap.add_argument("--max_train_docs", type=int, default=None, help="smoke tests")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true", help="save the trainable weights of the best evaluation")
    ap.add_argument("--inference", action="store_true", help="no training: load <out>/classifier.pt and predict dev + test")
    ap.add_argument("--predict_train", action="store_true", help="also save logits of the train docs (KD teacher)")
    args = ap.parse_args()
    args.seq = args.seq or args.max_len

    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    def prep(name):
        ids, y, toks = D.load_tokens(name)
        ch = [D.doc_chunks(t, args.view, args.max_len, args.max_chunks, args.head_len) for t in toks]
        return ids, y, ch, [len(t) for t in toks]
    tr_ids, tr_y, tr_ch, tr_len = prep("train")
    if args.max_train_docs:
        sel = np.random.RandomState(0).permutation(len(tr_ids))[:args.max_train_docs]
        tr_ids, tr_y = [tr_ids[i] for i in sel], tr_y[sel]
        tr_ch, tr_len = [tr_ch[i] for i in sel], [tr_len[i] for i in sel]
    dv_ids, dv_y, dv_ch, dv_len = prep("dev_clean")
    te_ids, _, te_ch, te_len = prep("test")
    prior = np.bincount(tr_y, minlength=2) / len(tr_y)
    prior_thr = float(np.log(prior[1] / prior[0]))            # margin threshold that removes the train prior
    soft = None
    if args.kd:
        z = np.load(args.kd)
        soft = torch.softmax(torch.from_numpy(np.stack([z[i] for i in tr_ids]).astype(np.float32)) / args.kd_T, -1)
        print(f"KD teacher {args.kd}: T {args.kd_T}, alpha {args.kd_alpha}, teacher train acc "
              f"{float((soft.argmax(-1).numpy() == tr_y).mean()):.3f}", flush=True)

    backbone, cfg = load_backbone(args.ckpt, max(4096, args.max_len))
    if args.n_layers:
        backbone.blocks = backbone.blocks[:args.n_layers]
        print(f"student: first {args.n_layers} of {cfg.n_layers} blocks", flush=True)
    model = DocClassifier(backbone, 2, args.dropout, args.doc_pool, args.n_pos, args.mlp)
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
    weight = None
    if args.class_weight == "balanced":
        weight = torch.tensor(0.5 / prior, dtype=torch.float32, device=device)

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
          f"{total} total | prior {prior.round(3).tolist()} | dev {len(dv_ids)} test {len(te_ids)}", flush=True)

    if args.inference:
        model.load_state_dict(torch.load(os.path.join(args.out, "classifier.pt"), map_location="cpu"), strict=False)
        dv, te = predict(model, dv_ch, dv_len, args, device), predict(model, te_ch, te_len, args, device)
        np.savez(os.path.join(args.out, "dev_logits.npz"), **dict(zip(dv_ids, dv)))
        np.savez(os.path.join(args.out, "test_logits.npz"), **dict(zip(te_ids, te)))
        thr = tune_threshold(dv_y, dv[:, 1] - dv[:, 0])
        write_submission(te_ids, (te[:, 1] - te[:, 0] > thr).astype(int), args.out)
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
                y = torch.from_numpy(tr_y[b["docs"]]).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(b)
                logits = logits.float()
                loss = F.cross_entropy(logits, y, weight=weight, reduction="none")
                if soft is not None:
                    pt = soft[b["docs"]].to(device)
                    ls = (logits / args.kd_T).log_softmax(-1)
                    kl = (pt * (pt.clamp_min(1e-12).log() - ls)).sum(-1)
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
                m = dv[:, 1] - dv[:, 0]
                thr = tune_threshold(dv_y, m)
                rec = {"step": step, "epoch": round(step / steps_epoch, 2), "train_loss": round(tot_loss / max(1, n_loss), 5),
                       "time_min": round((time.time() - t0) / 60, 1),
                       "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 1)}
                for k, t in (("argmax", 0.0), ("prior", prior_thr), ("tuned", thr)):
                    rec.update({f"{k}_{n}": round(v, 4) for n, v in scores(dv_y, m, t).items()})
                rec["thr"] = round(thr, 4)
                if rec["tuned_f1"] > best:
                    best = rec["tuned_f1"]
                    rec["best"] = True
                    te = predict(model, te_ch, te_len, args, device)
                    np.savez(os.path.join(args.out, "dev_logits.npz"), **dict(zip(dv_ids, dv)))
                    np.savez(os.path.join(args.out, "test_logits.npz"), **dict(zip(te_ids, te)))
                    json.dump({**rec, "thr": thr}, open(os.path.join(args.out, "best.json"), "w"), indent=1)
                    write_submission(te_ids, (te[:, 1] - te[:, 0] > thr).astype(int), args.out)
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
        np.savez(os.path.join(args.out, "train_logits.npz"), **dict(zip(tr_ids, tr)))
    print(f"done in {(time.time() - t0) / 60:.0f} min; best dev macro-F1 (tuned threshold) {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
