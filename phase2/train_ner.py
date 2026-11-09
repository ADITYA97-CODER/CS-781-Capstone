"""Fine-tune a Phase-1 model for L-NER (BIO token tagging) and evaluate / predict.

  # cross-validation half: train on fold1, select the epoch on fold2
  python train_ner.py --train fold1 --dev fold2 --out runs_p2/e_full_cv1 --gpu 0
  # final model: train on both folds for a fixed number of epochs, predict the test fold
  python train_ner.py --train fold1 fold2 --epochs 12 --predict fold3 --out runs_p2/e_full_all --gpu 1
  # FLERT-style windows (BIO head): sentence units of <= 256 tokens with 64 context tokens each side
  python train_ner.py --train fold1 --dev fold2 --windows flert --ctx 64 --unit_len 256 --out runs_p2/e_flert_cv1
  # knowledge distillation from teacher logits (kd_teacher.py): alpha * CE(gold) + (1 - alpha) * T^2 * KL
  python train_ner.py --train fold1 fold2 --kd runs_p2/kd_teacher_bio4.npz --kd_T 4 --kd_alpha 0.5 --predict fold3 --out runs_p2/e_kd

Outputs in --out: config.json, log.jsonl (one line per epoch), dev_pred.json and dev_scores.json for the
best epoch, test_logp.npz / prediction.json when --predict is given, and tagger_best.pt (with --save).
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

import ner_data as D
from ner_model import CRF, Tagger, load_backbone
from peft import add_adapters
import span_head as S

ROOT = D.ROOT


def prepare(split_names, tok):
    texts, gold = {}, {}
    for name in split_names:
        t, g = D.load_split(name)
        texts.update(t)
        if g is not None:
            gold.update(g)
    docs = []
    for k, text in texts.items():
        ids, offs = D.encode_doc(tok, text)
        tags = D.bio_tags(offs, gold[k]) if k in gold else None
        spans = D.gold_token_spans(offs, gold[k]) if k in gold else None
        docs.append((k, ids, offs, tags, spans))
    return texts, (gold or None), docs


@torch.no_grad()
def predict_logp(model, win, device, batch_size):
    """-> list of (seq_len, n_tags) float32 log-prob arrays, one per window row."""
    model.eval()
    out = []
    for i in range(0, len(win), batch_size):
        ids = win.ids[i:i + batch_size].to(device)
        valid = win.valid[i:i + batch_size].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, valid)
        out.extend(logits.float().log_softmax(-1).cpu().numpy())
    model.train()
    return out


def decode_all(texts, docs, win, row_logp):
    merged = D.merge_window_scores([len(d[1]) for d in docs], win.meta, row_logp)
    pred = {d[0]: D.decode_doc(texts[d[0]], d[2], lp) for d, lp in zip(docs, merged)}
    return pred, merged


def span_decode_all(texts, docs, cands, threshold):
    return {d[0]: S.decode_doc_spans(texts[d[0]], d[2], c, threshold) for d, c in zip(docs, cands)}


def span_candidates(model, win, docs, device, batch_size):
    rows = S.predict_candidates(model, win, device, batch_size)
    return S.merge_candidates([len(d[1]) for d in docs], win.meta, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs/sft_archE_v2_upweight/ckpt_best.pt"))
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--predict", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_frac", type=float, default=0.06)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--header_len", type=int, default=128)
    ap.add_argument("--content_len", type=int, default=1024)
    ap.add_argument("--train_stride", type=int, default=512)
    ap.add_argument("--eval_stride", type=int, default=768)
    ap.add_argument("--causal", action="store_true", help="keep the pretraining causal mask")
    ap.add_argument("--peft", default="none", choices=["none", "lorma", "lora", "bitfit", "ia3"],
                    help="freeze the backbone and train adapters (+ head) instead of full fine-tuning")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--head", default="bio", choices=["bio", "span"])
    ap.add_argument("--max_span", type=int, default=48, help="span head: longest candidate in tokens")
    ap.add_argument("--span_hidden", type=int, default=256)
    ap.add_argument("--span_threshold", type=float, default=0.5, help="span head: test-time decision threshold")
    ap.add_argument("--span_neg_ratio", type=float, default=20.0,
                    help="span head: total loss weight of O candidates per batch = ratio x weight of gold spans "
                         "(plain CE, ~1900 O per gold span, collapsed to all-O)")
    ap.add_argument("--windows", default="doc", choices=["doc", "flert"],
                    help="doc: long overlapping windows; flert: one unit (sentences) + --ctx tokens each side")
    ap.add_argument("--ctx", type=int, default=64, help="flert: context tokens on each side of a unit")
    ap.add_argument("--unit_len", type=int, default=256, help="flert: longest unit in tokens")
    ap.add_argument("--no_pack", action="store_true", help="flert: one sentence per unit (no packing)")
    ap.add_argument("--ctx_dropout", type=float, default=0.0,
                    help="flert: probability of hiding a training row's context (header kept)")
    ap.add_argument("--crf", action="store_true", help="linear-chain CRF on top of the BIO logits (NLL loss)")
    ap.add_argument("--crf_lr", type=float, default=1e-2, help="learning rate of the CRF transition scores")
    ap.add_argument("--crf_ce", type=float, default=0.0,
                    help="CRF: weight of an added token cross-entropy (CRF NLL alone diverged after epoch 6, 11 Oct)")
    ap.add_argument("--kd", default=None, help="knowledge distillation: teacher logits npz from kd_teacher.py")
    ap.add_argument("--kd_T", type=float, default=4.0, help="KD temperature")
    ap.add_argument("--kd_alpha", type=float, default=0.5, help="KD: weight of the gold-label CE (1 - alpha on the KL)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true", help="save the best (or final) tagger weights")
    ap.add_argument("--load", default=None, help="tagger weights to load (with --epochs 0: predict only)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    assert args.windows == "doc" or args.head == "bio", "flert windows are implemented for the BIO head only"
    assert args.kd is None or args.head == "bio", "knowledge distillation is implemented for the BIO head only"
    assert not args.crf or (args.head == "bio" and args.kd is None), "CRF: BIO head only, without KD"

    def windows(texts, docs, stride):
        if args.windows == "flert":
            bounds = [D.sentence_bounds(texts[d[0]], d[2]) for d in docs]
            return D.FlertWindows(docs, bounds, args.header_len, args.ctx, args.unit_len, pack=not args.no_pack)
        return D.DocWindows(docs, args.header_len, args.content_len, stride)

    tok = D.tokenizer()
    tr_texts, tr_gold, tr_docs = prepare(args.train, tok)
    D.learn_trims(tr_texts, tr_gold)                       # offset trims from the training folds only
    tr_win = windows(tr_texts, tr_docs, args.train_stride)
    soft = None
    if args.kd:
        # teacher soft targets per row: mean over teachers of softmax(z / T), at the labelled positions
        tz = np.load(args.kd)
        soft = torch.zeros(len(tr_win), tr_win.seq_len, len(D.TAGS), dtype=torch.float16)
        for r, (di, lo, hi, c0) in enumerate(tr_win.meta):
            z = torch.from_numpy(tz[tr_docs[di][0]][:, lo:hi].astype(np.float32))
            soft[r, c0:c0 + hi - lo] = (z / args.kd_T).softmax(-1).mean(0).half()
        assert ((soft.sum(-1) > 0) == (tr_win.labels >= 0)).all()
        print(f"KD: {tz[tr_docs[0][0]].shape[0]} teachers from {args.kd}, T {args.kd_T}, alpha {args.kd_alpha}", flush=True)
    dev = None
    if args.dev:
        dv_texts, dv_gold, dv_docs = prepare([args.dev], tok)
        dev = (dv_texts, dv_gold, dv_docs, windows(dv_texts, dv_docs, args.eval_stride))

    backbone, cfg = load_backbone(args.ckpt)
    model = Tagger(backbone, len(D.TAGS), bidirectional=not args.causal, dropout=args.dropout)
    if args.head == "span":
        model.head = S.SpanHead(cfg.d_model, args.max_span, args.span_hidden, args.dropout)
    if args.crf:
        model.crf = CRF(len(D.TAGS), D._ALLOWED)
    c0 = args.header_len + 1
    thresholds = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95)
    if args.peft != "none":
        n_wrapped = add_adapters(model.backbone, args.peft, args.rank, args.alpha)
        print(f"{args.peft}: {n_wrapped} linear layers adapted, rank {args.rank}, alpha {args.alpha}", flush=True)
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location="cpu"))
        print(f"loaded {args.load}", flush=True)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    head_params = list(model.head.parameters())
    decay, no_decay = [], []
    for name, p in model.backbone.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.dim() < 2 or "norm" in name or "lambda" in name or "subln" in name else decay).append(p)
    opt = torch.optim.AdamW([
        {"params": decay, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": no_decay, "lr": args.lr, "weight_decay": 0.0},
        {"params": head_params, "lr": args.head_lr, "weight_decay": 0.0},
    ] + ([{"params": list(model.crf.parameters()), "lr": args.crf_lr, "weight_decay": 0.0}] if args.crf else []), betas=(0.9, 0.98), eps=1e-6)
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_epoch = math.ceil(len(tr_win) / args.batch_size)
    total = steps_per_epoch * args.epochs
    warm = max(1, int(args.warmup_frac * total))

    def lr_scale(step):
        return step / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))

    log = open(os.path.join(args.out, "log.jsonl"), "a")
    print(f"params {n_params/1e6:.2f}M trainable {n_train/1e6:.2f}M | {args.windows} train windows {len(tr_win)} "
          f"of {tr_win.seq_len} tokens, {int((tr_win.labels >= 0).sum())} labelled "
          f"({steps_per_epoch} steps/epoch, {total} total) | dev windows {len(dev[3]) if dev else 0}", flush=True)
    best, step, t0 = -1.0, 0, time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(tr_win))
        tot_loss, n_b = 0.0, 0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            ids, valid, labels = tr_win.ids[idx].to(device), tr_win.valid[idx].to(device), tr_win.labels[idx].to(device)
            if args.ctx_dropout > 0:
                drop = torch.rand(len(idx), 1, device=device) < args.ctx_dropout
                valid = valid & ~(tr_win.ctx[idx].to(device) & drop)
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * lr_scale(step)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.head == "span":
                    logits, ok = model.head(model.hidden(ids, valid)[:, c0:], valid[:, c0:])
                    labels = S.span_labels([tr_win.spans[x] for x in idx.tolist()], None, args.max_span, ok).to(device)
                else:
                    logits = model(ids, valid)
            if args.head == "span":
                ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1),
                                     ignore_index=-100, reduction="none")
                lab = labels.view(-1)
                pos, neg = lab > 0, lab == 0
                n_pos, n_neg = pos.sum().clamp(min=1), neg.sum().clamp(min=1)
                w = args.span_neg_ratio * n_pos / n_neg
                loss = (ce[pos].sum() + w * ce[neg].sum()) / (n_pos + w * n_neg)
            else:
                if args.crf:
                    loss = model.crf.nll(logits.float(), labels)
                    if args.crf_ce > 0:
                        loss = loss + args.crf_ce * F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                                                                    labels.view(-1), ignore_index=-100)
                else:
                    loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                if soft is not None:
                    # Hinton et al. (2015): alpha * CE(gold) + (1 - alpha) * T^2 * KL(teacher_T || student_T)
                    m = labels >= 0
                    pt = soft[idx].to(device).float()[m]
                    ls = (logits.float()[m] / args.kd_T).log_softmax(-1)
                    kl = (pt * (pt.clamp_min(1e-12).log() - ls)).sum(-1).mean()
                    loss = args.kd_alpha * loss + (1 - args.kd_alpha) * args.kd_T ** 2 * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            step += 1
            tot_loss += loss.item()
            n_b += 1
        rec = {"epoch": epoch, "step": step, "train_loss": tot_loss / n_b, "time_s": round(time.time() - t0, 1),
               "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2)}
        if args.crf:
            D.set_crf(model.crf.export())
            np.savez(os.path.join(args.out, "crf_last.npz"), *model.crf.export())
            rec["crf_trans_absmax"] = round(float(model.crf.trans.abs().max()), 3)
        if dev:
            dv_texts, dv_gold, dv_docs, dv_win = dev
            if args.head == "span":
                cands = span_candidates(model, dv_win, dv_docs, device, args.batch_size)
                by_thr = {t: span_decode_all(dv_texts, dv_docs, cands, t) for t in thresholds}
                f1_thr = {t: D.strict_scores(dv_gold, pr)[0] for t, pr in by_thr.items()}
                rec["dev_f1_by_threshold"] = {str(t): round(v, 5) for t, v in f1_thr.items()}
                pred = by_thr[args.span_threshold]       # epoch selection at the fixed test threshold
            else:
                pred, dv_merged = decode_all(dv_texts, dv_docs, dv_win,
                                             predict_logp(model, dv_win, device, args.batch_size * 2))
            f1, p, r, per = D.strict_scores(dv_gold, pred)
            rec.update({"dev_f1": round(f1, 5), "dev_p": round(p, 5), "dev_r": round(r, 5),
                        "per_label_f1": {l: round(v[2], 4) for l, v in per.items()}})
            if f1 > best:
                best = f1
                rec["best"] = True
                json.dump(pred, open(os.path.join(args.out, "dev_pred.json"), "w"))
                json.dump({"epoch": epoch, "f1": f1, "p": p, "r": r,
                           "per_label": {l: dict(zip(["p", "r", "f1", "tp", "fp", "fn"], v)) for l, v in per.items()}},
                          open(os.path.join(args.out, "dev_scores.json"), "w"), indent=1)
                if args.head == "span":
                    S.save_candidates(os.path.join(args.out, "dev_cands.npz"), [d[0] for d in dv_docs], cands)
                else:
                    np.savez_compressed(os.path.join(args.out, "dev_logp.npz"),
                                        **{d[0]: lp.astype(np.float16) for d, lp in zip(dv_docs, dv_merged)})
                if args.crf:
                    np.savez(os.path.join(args.out, "crf.npz"), *model.crf.export())
                if args.save:
                    torch.save(model.state_dict(), os.path.join(args.out, "tagger_best.pt"))
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    if args.save and not dev and not args.load:
        torch.save(model.state_dict(), os.path.join(args.out, "tagger_final.pt"))
    if args.crf:
        D.set_crf(model.crf.export())          # also after --load (the CRF is part of the tagger state)
        if not dev:
            np.savez(os.path.join(args.out, "crf.npz"), *model.crf.export())
    if args.predict:
        te_texts, _, te_docs = prepare([args.predict], tok)
        te_win = windows(te_texts, te_docs, args.eval_stride)
        if args.head == "span":
            cands = span_candidates(model, te_win, te_docs, device, args.batch_size)
            pred = span_decode_all(te_texts, te_docs, cands, args.span_threshold)
            S.save_candidates(os.path.join(args.out, "test_cands.npz"), [d[0] for d in te_docs], cands)
        else:
            pred, merged = decode_all(te_texts, te_docs, te_win, predict_logp(model, te_win, device, args.batch_size * 2))
            np.savez_compressed(os.path.join(args.out, "test_logp.npz"), **{d[0]: lp for d, lp in zip(te_docs, merged)})
        json.dump(pred, open(os.path.join(args.out, "prediction.json"), "w"))
        print(f"wrote {len(pred)} test docs, {sum(map(len, pred.values()))} entities", flush=True)
    print(f"done in {time.time() - t0:.0f}s; best dev F1 {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
