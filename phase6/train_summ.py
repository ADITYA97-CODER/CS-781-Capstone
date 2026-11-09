"""Fine-tune a Phase-1 decoder (Arch E / F) for abstractive summarization (decoder-only: document view, prompt, summary).

  # full fine-tuning of Arch E (SFT upweight), head 3072 + tail 1024 document tokens, length tag in the prompt
  python train_summ.py --out runs_p6/e_full_ht4k --gpu 1
  # LoRMA+ / LoRA / Houlsby adapters / BitFit / IA3 / partial FT
  python train_summ.py --peft lorma --lr 5e-4 --out runs_p6/e_lorma_ht4k --gpu 1
  # KD: 10-layer student, token-level KL to a trained teacher run + gold CE
  python train_summ.py --n_layers 10 --kd runs_p6/e_full_ht4k --out runs_p6/kd_e10 --gpu 1

Selection: summary-token loss on the 300 held-out dev docs (evaluated evals_per_epoch times per epoch); the trainable
weights of the best evaluation are saved to --out/weights.pt. Generation and scoring: gen_summ.py.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import summ_data as D
from summ_model import Summarizer, add_bottleneck_adapters, load_lm

sys.path.insert(0, os.path.join(D.ROOT, "phase2"))
from peft import add_adapters  # noqa: E402

DEFAULT_CKPT = {"E": os.path.join(D.ROOT, "runs/sft_archE_v2_upweight/ckpt_best.pt"),
                "F": os.path.join(D.ROOT, "runs/archF_nodiff_145M/ckpt_best.pt")}


def apply_technique(bb, a):
    """Applies peft / partial FT / student truncation of run config `a` (a dict) to backbone bb."""
    if a.get("n_layers"):
        bb.blocks = bb.blocks[:a["n_layers"]]
    p = a.get("peft", "none")
    if p in ("houlsby", "pfeiffer"):
        return add_bottleneck_adapters(bb, p, a.get("adapter_size", 64))
    if p != "none":
        return add_adapters(bb, p, a.get("rank", 8), a.get("alpha", 16))
    if a.get("train_layers") is not None:
        nb = len(bb.blocks)
        for name, prm in bb.named_parameters():
            prm.requires_grad = (name.startswith("blocks.") and int(name.split(".")[1]) >= nb - a["train_layers"]) \
                or name.startswith("norm.")
    return 0


def load_trained(run_dir, device, max_len=8192):
    """Backbone of a finished run (base checkpoint + technique + saved weights), eval mode, on device."""
    a = json.load(open(os.path.join(run_dir, "config.json")))
    bb, _ = load_lm(a["ckpt"], max_len)
    apply_technique(bb, a)
    sd = torch.load(os.path.join(run_dir, "weights.pt"), map_location="cpu")
    missing = bb.load_state_dict(sd, strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys[:5]
    return bb.to(device).eval(), a


@torch.no_grad()
def val_loss(model, ex, args, device):
    model.eval()
    tot, n = 0.0, 0
    for mb in D.pack(ex, list(range(len(ex))), args.rows, args.seq):
        ids, lab = D.collate(mb, ex, args.rows, args.seq, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, k, _, _ = model(ids, lab)
        tot += loss.item()
        n += k
    model.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="E", choices=["E", "F"])
    ap.add_argument("--ckpt", default=None, help="default: Arch E SFT upweight / Arch F base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--view", default="headtail", choices=["head", "headtail", "spread"])
    ap.add_argument("--doc_len", type=int, default=4096, help="document tokens read")
    ap.add_argument("--head_len", type=int, default=3072, help="headtail: tokens from the start")
    ap.add_argument("--sum_len", type=int, default=1536, help="summary tokens trained on (longer ones are cut)")
    ap.add_argument("--len_tag", type=int, default=1, help="target length (words, rounded to 50) in the prompt")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--seq", type=int, default=6144)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--evals_per_epoch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--peft", default="none", choices=["none", "lorma", "lora", "rslora", "bitfit", "ia3",
                                                       "houlsby", "pfeiffer"])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--adapter_size", type=int, default=64)
    ap.add_argument("--train_layers", type=int, default=None, help="partial FT: top k blocks (+ final norm)")
    ap.add_argument("--n_layers", type=int, default=None, help="student: keep only the first n_layers blocks")
    ap.add_argument("--kd", default=None, help="teacher run dir (token-level KD)")
    ap.add_argument("--kd_T", type=float, default=1.0)
    ap.add_argument("--kd_alpha", type=float, default=0.5, help="weight of the gold CE (1 - alpha on T^2 KL)")
    ap.add_argument("--max_train_docs", type=int, default=None)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.ckpt = args.ckpt or DEFAULT_CKPT[args.arch]

    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    _, tr_d, tr_s, tr_w, _ = D.load_tokens("train")
    _, dv_d, dv_s, dv_w, _ = D.load_tokens("dev")
    if args.max_train_docs:
        sel = np.random.RandomState(0).permutation(len(tr_d))[:args.max_train_docs]
        tr_d, tr_s, tr_w = [tr_d[i] for i in sel], [tr_s[i] for i in sel], tr_w[sel]
    tr_ex = D.build_examples(tr_d, tr_s, tr_w, args)
    dv_ex = D.build_examples(dv_d, dv_s, dv_w, args)

    bb, cfg = load_lm(args.ckpt, args.seq + 64)
    n_adapted = apply_technique(bb, vars(args))
    model = Summarizer(bb, args.dropout).to(device)
    teacher = None
    if args.kd:
        teacher_bb, ta = load_trained(args.kd, device, args.seq + 64)
        teacher = Summarizer(teacher_bb).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"KD teacher {args.kd} ({len(teacher_bb.blocks)} blocks, {ta.get('peft')}), T {args.kd_T}, "
              f"alpha {args.kd_alpha}", flush=True)
    if not args.no_compile:
        from torch import _dynamo
        _dynamo.config.cache_size_limit = 64
        _dynamo.config.automatic_dynamic_shapes = False
        model.set_compiled(torch.compile(model.decoder, dynamic=False))
        if teacher is not None:
            teacher.set_compiled(torch.compile(teacher.decoder, dynamic=False))
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.dim() < 2 or "norm" in name or "lambda" in name or "subln" in name else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.98), eps=1e-6)

    mb_epoch = len(D.pack(tr_ex, list(range(len(tr_ex))), args.rows, args.seq))
    steps_epoch = math.ceil(mb_epoch / args.accum)
    total = int(steps_epoch * args.epochs)
    warm = max(1, int(args.warmup_frac * total))
    eval_at = {min(total, int(round(steps_epoch * k / args.evals_per_epoch)))
               for k in range(1, int(math.ceil(args.epochs * args.evals_per_epoch)) + 1)}
    tr_tok = sum(len(x) for x, _ in tr_ex)
    tr_lab = sum(int((y != -100).sum()) for _, y in tr_ex)
    print(f"{args.arch} {args.peft} ({n_adapted} adapted) | {args.view} doc {args.doc_len} sum {args.sum_len} "
          f"len_tag {args.len_tag} | params {n_params/1e6:.2f}M trainable {n_train/1e6:.2f}M | train {len(tr_ex)} docs "
          f"{tr_tok/1e6:.1f}M tok ({tr_lab/1e6:.1f}M labelled), {mb_epoch} mbs of {args.rows}x{args.seq} "
          f"({tr_tok / (mb_epoch * args.rows * args.seq):.0%} filled) | {steps_epoch} steps/epoch, {total} total",
          flush=True)

    log = open(os.path.join(args.out, "log.jsonl"), "a")
    v0 = val_loss(model, dv_ex, args, device)
    print(json.dumps({"step": 0, "val_loss": round(v0, 4)}), flush=True)
    log.write(json.dumps({"step": 0, "val_loss": v0}) + "\n")
    best, step, t0, seen = float("inf"), 0, time.time(), 0
    model.train()
    while step < total:
        mbs = D.pack(tr_ex, torch.randperm(len(tr_ex)).tolist(), args.rows, args.seq)
        tot, ntok, tot_kl = 0.0, 0, 0.0
        for i in range(0, len(mbs), args.accum):
            if step >= total:
                break
            lr = args.lr * (step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
            for g in opt.param_groups:
                g["lr"] = lr
            group = mbs[i:i + args.accum]
            batch = [D.collate(mb, tr_ex, args.rows, args.seq, device) for mb in group]
            n_group = sum(int((lab != -100).sum()) for _, lab in batch)
            for ids, lab in batch:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ce, k, logits, sel = model(ids, lab)
                    loss = ce
                    if teacher is not None:
                        with torch.no_grad():
                            tl = teacher.logits_at(teacher.hidden(ids), sel)
                        pt = (tl / args.kd_T).softmax(-1)
                        kl = (pt * (pt.clamp_min(1e-9).log() - (logits / args.kd_T).log_softmax(-1))).sum()
                        loss = args.kd_alpha * ce + (1 - args.kd_alpha) * args.kd_T ** 2 * kl
                        tot_kl += kl.item()
                (loss / n_group).backward()
                tot += ce.item()
                ntok += k
                seen += int((ids != 0).sum())
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                el = time.time() - t0
                print(f"step {step}/{total} ce {tot / ntok:.4f}" + (f" kl {tot_kl / ntok:.4f}" if teacher else "") +
                      f" lr {lr:.2e} {seen / el / 1e3:.1f}k tok/s {el / 60:.1f} min, eta {(total - step) * el / step / 60:.0f} min",
                      flush=True)
            if step in eval_at:
                v = val_loss(model, dv_ex, args, device)
                rec = {"step": step, "epoch": round(step / steps_epoch, 2), "train_ce": round(tot / max(1, ntok), 4),
                       "val_loss": round(v, 4), "time_min": round((time.time() - t0) / 60, 1),
                       "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 1)}
                if v < best:
                    best = v
                    rec["best"] = True
                    keep = {n for n, p in model.backbone.named_parameters() if p.requires_grad}
                    sd = model.backbone.state_dict()
                    torch.save({k: t for k, t in sd.items() if k in keep or args.peft == "none"},
                               os.path.join(args.out, "weights.pt"))
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
    print(f"done in {(time.time() - t0) / 60:.0f} min; best dev loss {best:.4f} (step 0: {v0:.4f})", flush=True)


if __name__ == "__main__":
    main()
