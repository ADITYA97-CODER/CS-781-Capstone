"""Fine-tune a Phase-1 model for Rhetorical Role prediction (one label per sentence) and evaluate / predict.

  # select the epoch on the official dev split (CL_dev + IT_dev), then predict the test docs with the best epoch
  python train_rr.py --out runs_p3/e_full_ctx --gpu 0 --predict
  # sentence-only baseline (no context: one sentence per row)
  python train_rr.py --ctx none --out runs_p3/e_full_noctx
  # PEFT: freeze the backbone and train adapters + head (phase2/peft.py)
  python train_rr.py --peft lorma --rank 8 --alpha 16 --lr 5e-4 --out runs_p3/e_lorma
  # 2-fold CV over all 90 labelled docs (run both halves, then cv_score.py --runs <out0> <out1>)
  python train_rr.py --train cv2_f0 --dev cv2_f1 --epoch_logp --out runs_p3/cv/<name>_f0
  python train_rr.py --train cv2_f1 --dev cv2_f0 --epoch_logp --out runs_p3/cv/<name>_f1
  # Laya-style head: a learned marker token after every sentence, MLP on its hidden state, logit-adjusted CE
  python train_rr.py --head laya --la_tau 1 --out runs_p3/e_full_laya
  # the same without markers: mean-pooled sentence vector + Laya MLP + logit-adjusted CE
  python train_rr.py --head pool --mlp --la_tau 1 --out runs_p3/e_full_laya_nomarker

Outputs in --out: config.json, log.jsonl (one line per epoch), dev_scores.json / dev_logp.npz for the best epoch,
and with --predict: test_logp.npz, prediction.json and submission.zip (prediction.json at the zip root).
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

import rr_data as D
from rr_model import LayaClassifier, SentClassifier, load_backbone

sys.path.insert(0, os.path.join(D.ROOT, "phase2"))
from peft import add_adapters  # noqa: E402

TRAIN = ["CL_train", "IT_train"]
DEV = ["CL_dev", "IT_dev"]
TEST = ["CL_test", "IT_test"]


def batch(win, idx, device):
    """Rows idx -> device tensors cut to the longest row of the batch (rows are left-aligned, padding is masked out
    and RoPE positions start at 0, so the cut does not change the output)."""
    L = int(win.valid[idx].sum(1).max())
    return (win.ids[idx, :L].to(device), win.valid[idx, :L].to(device), win.sid[idx, :L].to(device),
            win.pos[idx].to(device), win.dom[idx].to(device), win.n_slots, win.mpos[idx].to(device))


def length_batches(win, batch_size, shuffle=True, chunk=50):
    """Batches of row indices. shuffle: random order, then rows sorted by length inside chunks of chunk x batch_size
    rows and the batches shuffled (less padding for short rows); else all rows sorted by length (prediction)."""
    lens = win.valid.sum(1)
    if not shuffle:
        order = torch.argsort(lens)
        return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    order = torch.randperm(len(win))
    out = []
    for c in range(0, len(order), chunk * batch_size):
        part = order[c:c + chunk * batch_size]
        part = part[torch.argsort(lens[part])]
        out += [part[i:i + batch_size] for i in range(0, len(part), batch_size)]
    return [out[i] for i in torch.randperm(len(out)).tolist()]


@torch.no_grad()
def predict_logp(model, win, docs, device, batch_size):
    model.eval()
    rows = [None] * len(win)
    for idx in length_batches(win, batch_size, shuffle=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(*batch(win, idx, device))
        for r, lp in zip(idx.tolist(), logits.float().log_softmax(-1).cpu().numpy()):
            rows[r] = lp
    model.train()
    return D.merge_scores(docs, win, rows)


def evaluate(docs, logp, offset=None):
    """offset (K,): added to the log-probs before the argmax (logit adjustment: tau * log prior puts the priors back)."""
    if offset is not None:
        logp = [lp + offset for lp in logp]
    y = np.concatenate([d["y"] for d in docs])
    p = np.concatenate([lp.argmax(-1) for lp in logp])
    f1, prec, rec, per = D.macro_scores(y, p)
    out = {"f1": f1, "p": prec, "r": rec, "acc": float((y == p).mean()),
           "per_label": {l: dict(zip(["p", "r", "f1", "n"], v)) for l, v in per.items()}}
    present = [v[2] for v in per.values() if v[3] > 0]
    out["f1_present"] = float(np.mean(present))
    for dom in D.DOMAINS:
        sel = [i for i, d in enumerate(docs) if d["domain"] == dom]
        if not sel:
            continue
        yd = np.concatenate([docs[i]["y"] for i in sel])
        pd = np.concatenate([logp[i].argmax(-1) for i in sel])
        out[f"f1_{dom}"] = D.macro_scores(yd, pd)[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(D.ROOT, "runs/sft_archE_v2_upweight/ckpt_best.pt"))
    ap.add_argument("--train", nargs="+", default=TRAIN)
    ap.add_argument("--dev", nargs="*", default=DEV)
    ap.add_argument("--predict", action="store_true", help="predict CL_test + IT_test with the best (or last) epoch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--stop_epoch", type=int, default=None,
                    help="stop after this epoch, keeping the --epochs lr schedule (replays a dev-selected epoch without dev)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_frac", type=float, default=0.06)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--ctx", default="window", choices=["window", "none"])
    ap.add_argument("--max_len", type=int, default=1024, help="window: tokens per row")
    ap.add_argument("--train_stride", type=int, default=512, help="window: tokens between training row starts")
    ap.add_argument("--eval_stride", type=int, default=512)
    ap.add_argument("--max_sent_tok", type=int, default=192)
    ap.add_argument("--bucket", action="store_true", help="training batches of similar-length rows (for --ctx none)")
    ap.add_argument("--n_pos", type=int, default=16, help="relative-position buckets (<= 1: no position embedding)")
    ap.add_argument("--domain_emb", action="store_true")
    ap.add_argument("--class_weight", default="none", choices=["none", "sqrt", "inv"],
                    help="CE weights from training label counts: (n_max / n_c) ** 0.5 or n_max / n_c (capped at 50)")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--head", default="pool", choices=["pool", "laya"],
                    help="pool: mean of the sentence's tokens + linear; laya: marker token after every sentence + MLP")
    ap.add_argument("--mlp", action="store_true", help="pool head: the Laya MLP (d -> GELU -> K) instead of a linear")
    ap.add_argument("--la_tau", type=float, default=0.0,
                    help="logit adjustment (Menon et al. 2021): train on logits + tau * log(prior), predict with the raw "
                         "logits (balanced); dev also reports f1_prior = argmax with the priors put back")
    ap.add_argument("--causal", action="store_true", help="keep the pretraining causal mask")
    ap.add_argument("--peft", default="none", choices=["none", "lorma", "lora", "rslora", "bitfit", "ia3"])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--n_layers", type=int, default=None,
                    help="student size: keep only the first n_layers blocks of the Phase-1 model")
    ap.add_argument("--kd", nargs="+", default=None,
                    help="knowledge distillation from out-of-fold CV predictions: <cv run prefix>:<epoch> ..., e.g. "
                         "../runs_p3/cv/ctx_win1024:17 (reads <prefix>_f0/_f1/dev_logp_ep<epoch>.npz); teacher = mean "
                         "of the configs' softmax(log p / T)")
    ap.add_argument("--kd_T", type=float, default=2.0)
    ap.add_argument("--kd_alpha", type=float, default=0.5, help="KD: weight of the gold CE (1 - alpha on T^2 * KL)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoch_logp", action="store_true",
                    help="save the dev log-probs of every epoch (dev_logp_ep<e>.npz), for cv_score.py")
    ap.add_argument("--epoch_test_logp", action="store_true",
                    help="also save the test log-probs of every epoch (test_logp_ep<e>.npz), e.g. to ensemble CV halves")
    ap.add_argument("--save", action="store_true", help="save the best (or final) classifier weights")
    ap.add_argument("--load", default=None, help="classifier weights to load (with --epochs 0: predict only)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    tok = D.tokenizer()
    load = lambda names: D.encode_docs(tok, [d for n in names for d in D.load_split(n)], args.max_sent_tok)  # noqa: E731
    tr_docs = load(args.train)
    marker = args.head == "laya"
    tr_win = D.SentWindows(tr_docs, args.ctx, args.max_len, args.train_stride, args.n_pos, marker)
    soft = None
    if args.kd:
        # teacher probabilities per sentence (out-of-fold: each doc scored by the CV half that did not train on it)
        teach = {}
        for spec in args.kd:
            prefix, ep = spec.rsplit(":", 1)
            for f in (0, 1):
                z = np.load(f"{prefix}_f{f}/dev_logp_ep{ep}.npz")
                for k in z.files:
                    p = torch.softmax(torch.from_numpy(z[k].astype(np.float32)) / args.kd_T, -1)
                    teach.setdefault(k, []).append(p)
        teach = {k: torch.stack(v).mean(0) for k, v in teach.items()}
        assert all(len(teach[d["id"]]) == len(d["ids"]) for d in tr_docs), "teacher missing a training doc"
        soft = torch.zeros(len(tr_win), tr_win.n_slots, len(D.LABELS))
        for r, m in enumerate(tr_win.meta):
            for j, (di, i, _, _) in enumerate(m):
                soft[r, j] = teach[tr_docs[di]["id"]][i]
        print(f"KD: {len(args.kd)} teacher configs (out-of-fold), T {args.kd_T}, alpha {args.kd_alpha}", flush=True)
    dv_docs = load(args.dev) if args.dev else None
    if args.epoch_test_logp or args.predict:
        te_docs = load(TEST)
        te_win = D.SentWindows(te_docs, args.ctx, args.max_len, args.eval_stride, args.n_pos, marker)
    dv_win = D.SentWindows(dv_docs, args.ctx, args.max_len, args.eval_stride, args.n_pos, marker) if dv_docs else None

    counts = np.bincount(np.concatenate([d["y"] for d in tr_docs]), minlength=len(D.LABELS)).astype(np.float64)
    weight = None
    if args.class_weight != "none":
        w = counts.max() / np.maximum(counts, 1)
        w = np.minimum(w ** (0.5 if args.class_weight == "sqrt" else 1.0), 50.0)
        weight = torch.tensor(w / (w * counts).sum() * counts.sum(), dtype=torch.float32, device=device)
    offset = (args.la_tau * np.log(np.maximum(counts, 1) / counts.sum())).astype(np.float32) if args.la_tau else None
    offset_t = torch.tensor(offset, device=device) if offset is not None else None

    backbone, cfg = load_backbone(args.ckpt)
    if max(tr_win.seq_len, dv_win.seq_len if dv_win else 0) > backbone.rope_cos.size(0):    # rows longer than pretraining
        from models.layers import rope_tables
        cos, sin = rope_tables(cfg.head_dim, 4096, cfg.rope_theta)
        backbone.rope_cos, backbone.rope_sin = cos.to(backbone.rope_cos), sin.to(backbone.rope_sin)
        print(f"RoPE tables extended {cfg.max_seq_len} -> 4096", flush=True)
    if args.n_layers:
        backbone.blocks = backbone.blocks[:args.n_layers]
        print(f"student: first {args.n_layers} of {cfg.n_layers} blocks", flush=True)
    cls = LayaClassifier if args.head == "laya" else SentClassifier
    model = cls(backbone, len(D.LABELS), args.n_pos, args.domain_emb, not args.causal, args.dropout,
                **({"mlp": True} if args.mlp else {}))
    if args.peft != "none":
        n_wrapped = add_adapters(model.backbone, args.peft, args.rank, args.alpha)
        print(f"{args.peft}: {n_wrapped} linear layers adapted, rank {args.rank}, alpha {args.alpha}", flush=True)
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location="cpu"))
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    decay, no_decay = [], []
    for name, p in model.backbone.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.dim() < 2 or "norm" in name or "lambda" in name or "subln" in name else decay).append(p)
    opt = torch.optim.AdamW([
        {"params": decay, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": no_decay, "lr": args.lr, "weight_decay": 0.0},
        {"params": head_params, "lr": args.head_lr, "weight_decay": 0.0},
    ], betas=(0.9, 0.98), eps=1e-6)
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_epoch = math.ceil(len(tr_win) / args.batch_size)
    total = steps_per_epoch * args.epochs
    warm = max(1, int(args.warmup_frac * total))

    def lr_scale(step):
        return step / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))

    log = open(os.path.join(args.out, "log.jsonl"), "a")
    print(f"params {n_params/1e6:.2f}M trainable {n_train/1e6:.2f}M | ctx {args.ctx}: {len(tr_win)} train rows of "
          f"{tr_win.seq_len} tokens / {tr_win.n_slots} slots, {int((tr_win.y >= 0).sum())} labelled sentences "
          f"({sum(len(d['ids']) for d in tr_docs)} distinct) | {steps_per_epoch} steps/epoch, {total} total | "
          f"dev rows {len(dv_win) if dv_win else 0}", flush=True)
    best, best_state, step, t0 = -1.0, None, 0, time.time()
    model.train()
    for epoch in range(1, (args.stop_epoch or args.epochs) + 1):
        if args.bucket:
            batches = length_batches(tr_win, args.batch_size)
        else:
            order = torch.randperm(len(tr_win))
            batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        tot_loss, n_b = 0.0, 0
        for idx in batches:
            y = tr_win.y[idx].to(device)
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * lr_scale(step)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(*batch(tr_win, idx, device))
            logits = logits.float()
            if offset_t is not None:
                logits = logits + offset_t
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), weight=weight,
                                   ignore_index=-100, label_smoothing=args.label_smoothing)
            if soft is not None:
                # Hinton et al. (2015): alpha * CE(gold) + (1 - alpha) * T^2 * KL(teacher_T || student_T)
                m = y >= 0
                pt = soft[idx].to(device)[m]
                ls = (logits[m] / args.kd_T).log_softmax(-1)
                kl = (pt * (pt.clamp_min(1e-12).log() - ls)).sum(-1).mean()
                loss = args.kd_alpha * loss + (1 - args.kd_alpha) * args.kd_T ** 2 * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            step += 1
            tot_loss += loss.item()
            n_b += 1
        rec = {"epoch": epoch, "step": step, "train_loss": round(tot_loss / n_b, 5), "time_s": round(time.time() - t0, 1),
               "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2)}
        if dv_docs:
            logp = predict_logp(model, dv_win, dv_docs, device, args.batch_size * 2)
            sc = evaluate(dv_docs, logp)
            rec.update({k: round(v, 5) for k, v in sc.items() if k != "per_label"})
            rec["per_label_f1"] = {l: round(v["f1"], 3) for l, v in sc["per_label"].items()}
            if args.epoch_logp:
                np.savez_compressed(os.path.join(args.out, f"dev_logp_ep{epoch}.npz"),
                                    **{d["id"]: lp.astype(np.float16) for d, lp in zip(dv_docs, logp)})
            if args.epoch_test_logp:
                tlp = predict_logp(model, te_win, te_docs, device, args.batch_size * 2)
                np.savez_compressed(os.path.join(args.out, f"test_logp_ep{epoch}.npz"),
                                    **{d["id"]: lp.astype(np.float16) for d, lp in zip(te_docs, tlp)})
            if offset is not None:
                rec["f1_prior"] = round(evaluate(dv_docs, logp, offset)["f1"], 5)
            if sc["f1"] > best:
                best = sc["f1"]
                rec["best"] = True
                sc["epoch"] = epoch
                json.dump(sc, open(os.path.join(args.out, "dev_scores.json"), "w"), indent=1)
                np.savez_compressed(os.path.join(args.out, "dev_logp.npz"),
                                    **{d["id"]: lp.astype(np.float16) for d, lp in zip(dv_docs, logp)})
                trainable = {n for n, p in model.named_parameters() if p.requires_grad}
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                              if k in trainable or args.peft == "none"}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        print(f"restored best dev epoch (macro-F1 {best:.4f})", flush=True)
    if args.save:
        torch.save({k: v for k, v in model.state_dict().items()}, os.path.join(args.out, "classifier.pt"))
    if args.predict:
        logp = predict_logp(model, te_win, te_docs, device, args.batch_size * 2)
        np.savez_compressed(os.path.join(args.out, "test_logp.npz"), **{d["id"]: lp for d, lp in zip(te_docs, logp)})
        pred = {d["id"]: [D.LABELS[k] for k in lp.argmax(-1)] for d, lp in zip(te_docs, logp)}
        write_submission(pred, te_docs, args.out)
    print(f"done in {time.time() - t0:.0f}s; best dev macro-F1 {best:.4f}", flush=True)


def write_submission(pred, te_docs, out):
    """Validate against the official rules (ids, counts, label set) and write prediction.json + submission.zip."""
    assert set(pred) == {d["id"] for d in te_docs}
    for d in te_docs:
        assert len(pred[d["id"]]) == len(d["text"]), d["id"]
        assert all(l in D.LAB2ID for l in pred[d["id"]])
    path = os.path.join(out, "prediction.json")
    json.dump(pred, open(path, "w"))
    with zipfile.ZipFile(os.path.join(out, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, "prediction.json")
    print(f"wrote {len(pred)} test docs, {sum(map(len, pred.values()))} sentences -> {out}/submission.zip", flush=True)


if __name__ == "__main__":
    main()
