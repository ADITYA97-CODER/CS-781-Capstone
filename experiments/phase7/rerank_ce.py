"""Cross-encoder reranker for PCR on a Phase-1 decoder (Arch E / F): reads query and candidate together.

Pair input (one bidirectional segment, <= q_len + 2 * c_win tokens, RoPE 0..L-1):
  [query view: the <CITATION>-context sentences, first --q_len tokens] [sep]
  [candidate head: first --c_win tokens (SC header / headnote / opening)] [sep]
  [candidate evidence: the --c_win-token window (stride c_win/2) with the most query-term weight (idf of the
   query-view token ids, computed on the split's pool)]
Score = linear(mean of the pair's token states). The local (sliding-window) layers see neighbours, the global layers
let query and candidate tokens attend to each other.

Training: listwise. A group = 1 gold candidate + --n_neg negatives sampled from the query's top --neg_pool by the
stage-1 LTR (lexical + metadata, ltr.py --fit train, in-sample on train); softmax CE over the group. One group per
micro-batch (rows x seq packed); --accum groups per optimizer step. Epoch = every (query, gold) pair once.
Inference: the top --rerank candidates of the stage-1 LTR per query on dev / test -> scores saved to
runs_p7/<out>/ce_<split>.npz (qids, cids, S with -inf outside the reranked set) for ltr.py (--ce).
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
import torch.nn as nn
import torch.nn.functional as F

import pcr_data as D
from pcr_tok import load_tokens
from train_pcr import query_ctx_tokens

sys.path.insert(0, os.path.join(D.ROOT, "phase4"))
sys.path.insert(0, os.path.join(D.ROOT, "phase2"))
from cjpe_model import load_backbone, segment_masks  # noqa: E402
from peft import add_adapters  # noqa: E402

PAD_SEG = 1 << 20
SEP = 0          # eos id of the Phase-1 tokenizer, used as a separator


class CrossEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        backbone.lm_head = None
        for block in backbone.blocks:
            block.drop.p = 0.1
        self.window = backbone.cfg.window
        d = backbone.cfg.d_model
        self.head = nn.Linear(d, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        self.compiled = None

    def set_compiled(self, fn):
        object.__setattr__(self, "compiled", fn)

    def encoder(self, ids, pos, masks):
        bb = self.backbone
        x = bb.embed(ids)
        cos, sin = bb.rope_cos[pos], bb.rope_sin[pos]
        for block in bb.blocks:
            x = block(x, cos, sin, masks[block.attn.local])
        return bb.norm(x)

    def forward(self, b):
        masks = segment_masks(b["seg"], self.window)
        fn = self.encoder if self.compiled is None else self.compiled
        h = fn(b["ids"], b["pos"], masks).float()
        d = h.size(-1)
        tok = b["tok_pair"].view(-1)
        keep = tok >= 0
        n = b["n_pairs"]
        s = h.new_zeros(n, d).index_add_(0, tok[keep], h.view(-1, d)[keep])
        cnt = h.new_zeros(n).index_add_(0, tok[keep], h.new_ones(int(keep.sum())))
        return self.head(s / cnt.clamp(min=1)[:, None]).squeeze(-1)


class PairBuilder:
    def __init__(self, split, a):
        self.ids, self.toks = load_tokens(split)
        self.qt = query_ctx_tokens(split)
        self.a = a
        N = len(self.ids)
        df = np.zeros(32128, np.float64)
        for t in self.toks.values():
            df[np.unique(t)] += 1
        self.idf = np.log1p((N - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.idf[:300] = 0                                   # very frequent / special ids carry no signal
        self.qw = {}

    def qweights(self, q):
        if q not in self.qw:
            w = np.zeros_like(self.idf)
            u = np.unique(self.qt[q])
            w[u] = self.idf[u]
            self.qw[q] = w
        return self.qw[q]

    def pair(self, q, c):
        a = self.a
        qv = self.qt[q][:a.q_len]
        ct = self.toks[c]
        head = ct[:a.c_win]
        rest = ct[a.c_win:]
        if len(rest) > 0:
            w = self.qweights(q)[rest]
            cs = np.concatenate([[0], np.cumsum(w)])
            L = min(a.c_win, len(rest))
            starts = np.arange(0, max(1, len(rest) - L + 1), max(1, a.c_win // 2))
            best = starts[np.argmax(cs[starts + L] - cs[starts])]
            ev = rest[best:best + L]
        else:
            ev = rest
        return np.concatenate([qv, [SEP], head, [SEP], ev]).astype(np.int64)


def pack_pairs(seqs, rows, seq, device):
    ids = torch.zeros(rows, seq, dtype=torch.long)
    pos = torch.zeros(rows, seq, dtype=torch.long)
    segm = torch.full((rows, seq), PAD_SEG, dtype=torch.long)
    tok = torch.full((rows, seq), -1, dtype=torch.long)
    free = [0] * rows
    for k, s in enumerate(seqs):
        L = len(s)
        r = next(i for i in range(rows) if seq - free[i] >= L)
        o = free[r]
        ids[r, o:o + L] = torch.from_numpy(s)
        pos[r, o:o + L] = torch.arange(L)
        segm[r, o:o + L] = k
        tok[r, o:o + L] = k
        free[r] += L
    t = lambda x: x.to(device, non_blocking=True)  # noqa: E731
    return {"ids": t(ids), "pos": t(pos), "seg": t(segm), "tok_pair": t(tok), "n_pairs": len(seqs)}


def stage1_rankings(split, n):
    """Top-n candidate ids per query by the stage-1 LTR (lexical + metadata, fit on train). Cached."""
    path = os.path.join(D.ROOT, "runs_p7", "stage1", f"{split}_top{n}.json")
    if os.path.exists(path):
        return json.load(open(path))
    from ltr import split_matrices, pairs, predict_scores, fit
    qt, ct, Mt, Lt, SMt = split_matrices("train", [], "doc")
    _, gold_t, _ = D.load("train")
    X, y, _, names = pairs(range(len(qt)), Mt, Lt, 150, gold_t, qt, ct, SMt, True)
    model = fit(X, y, 1.0)
    out = {}
    for s in ("train", "dev", "test"):
        q, c, M, L, SM = split_matrices(s, [], "doc")
        sc = predict_scores(model, range(len(q)), M, L, max(150, n), SM, names)
        out[s] = {q[i]: [c[j] for j in o[np.argsort(-v, kind="stable")][:n]] for i, (o, v) in sc.items()}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(out[s], open(os.path.join(os.path.dirname(path), f"{s}_top{n}.json"), "w"))
    return out[split]


@torch.no_grad()
def score_split(model, pb, ranking, a, device):
    model.eval()
    items = [(q, c) for q, cs in ranking.items() for c in cs]
    seqs = [pb.pair(q, c) for q, c in items]
    order = sorted(range(len(items)), key=lambda i: -len(seqs[i]))
    per_mb = (a.rows * a.seq) // (a.q_len + 2 * a.c_win + 2)
    out = np.zeros(len(items), np.float32)
    for k in range(0, len(order), per_mb):
        idx = order[k:k + per_mb]
        b = pack_pairs([seqs[i] for i in idx], a.rows, a.seq, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[idx] = model(b).float().cpu().numpy()
    model.train()
    res = {}
    for (q, c), s in zip(items, out):
        res.setdefault(q, {})[c] = float(s)
    return res


def eval_dev(scores, ranking, gold):
    pred = {q: sorted(cs, key=lambda c: -scores[q][c])[:D.K] for q, cs in ranking.items()}
    return D.score(gold, pred)["f1"], D.score(gold, {q: cs[:D.K] for q, cs in ranking.items()})["f1"]


def save_scores(scores, split, out):
    qs, gold, pool = D.load(split)
    qids, cids = list(qs), list(pool)
    pos = {c: j for j, c in enumerate(cids)}
    S = np.full((len(qids), len(cids)), -np.inf, np.float32)
    for i, q in enumerate(qids):
        for c, s in scores.get(q, {}).items():
            S[i, pos[c]] = s
    np.savez(os.path.join(out, f"ce_{split}.npz"), qids=qids, cids=cids, S=S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(D.ROOT, "runs/sft_archE_v2_upweight/ckpt_best.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--q_len", type=int, default=1024)
    ap.add_argument("--c_win", type=int, default=512)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=2050)
    ap.add_argument("--n_neg", type=int, default=7)
    ap.add_argument("--neg_pool", type=int, default=30)
    ap.add_argument("--rerank", type=int, default=30)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--evals_per_epoch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--peft", default="lorma", choices=["none", "lorma", "lora", "ia3"])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--max_steps", type=int, default=None)
    a = ap.parse_args()
    assert a.rows * a.seq >= (a.n_neg + 1) * (a.q_len + 2 * a.c_win + 2), "a group must fit one micro-batch"
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device(f"cuda:{a.gpu}")
    torch.cuda.set_device(device)

    rk = {s: stage1_rankings(s, max(a.rerank, a.neg_pool)) for s in ("train", "dev", "test")}
    pbs = {s: PairBuilder(s, a) for s in ("train", "dev", "test")}
    _, gold_t, _ = D.load("train")
    _, gold_d, _ = D.load("dev")
    groups = [(q, g) for q, gs in gold_t.items() for g in gs]
    negs = {q: [c for c in rk["train"][q][:a.neg_pool] if c not in set(gold_t[q])] for q in gold_t}
    dv_rank = {q: v[:a.rerank] for q, v in rk["dev"].items()}
    te_rank = {q: v[:a.rerank] for q, v in rk["test"].items()}

    backbone, cfg = load_backbone(a.ckpt, max(4096, a.q_len + 2 * a.c_win + 2))
    model = CrossEncoder(backbone)
    if a.peft != "none":
        add_adapters(model.backbone, a.peft, a.rank, a.alpha)
    model = model.to(device)
    if not a.no_compile:
        from torch import _dynamo
        _dynamo.config.cache_size_limit = 64
        _dynamo.config.automatic_dynamic_shapes = False
        model.set_compiled(torch.compile(model.encoder, dynamic=False))
    bb = {id(p) for p in model.backbone.parameters()}
    opt = torch.optim.AdamW([{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": a.lr,
                              "weight_decay": 0.01},
                             {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in bb],
                              "lr": a.head_lr, "weight_decay": 0.0}], betas=(0.9, 0.98), eps=1e-6)
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_epoch = math.ceil(len(groups) / a.accum)
    total = a.max_steps or int(steps_epoch * a.epochs)
    warm = max(1, int(a.warmup_frac * total))
    eval_at = {min(total, int(round(steps_epoch * k / a.evals_per_epoch)))
               for k in range(1, int(math.ceil(a.epochs * a.evals_per_epoch)) + 1)} | {total}
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"cross-encoder | trainable {n_train / 1e6:.2f}M | {len(groups)} groups/epoch, {steps_epoch} steps/epoch, "
          f"{total} total | pair <= {a.q_len + 2 * a.c_win + 2} tokens", flush=True)
    s0, base_f1 = eval_dev({q: {c: -k for k, c in enumerate(cs)} for q, cs in dv_rank.items()}, dv_rank, gold_d)
    print(f"stage-1 LTR dev F1@7 {base_f1:.4f}", flush=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    best, step, t0 = -1.0, 0, time.time()
    order = []
    model.train()
    while step < total:
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * (step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
        tot = 0.0
        for _ in range(a.accum):
            if not order:
                order = random.sample(groups, len(groups))
            q, g = order.pop()
            cand = [g] + random.sample(negs[q], min(a.n_neg, len(negs[q])))
            b = pack_pairs([pbs["train"].pair(q, c) for c in cand], a.rows, a.seq, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = model(b).float()
            loss = F.cross_entropy(s[None], torch.zeros(1, dtype=torch.long, device=device)) / a.accum
            loss.backward()
            tot += loss.item()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 50 == 0:
            el = time.time() - t0
            print(f"step {step}/{total} loss {tot:.4f} {el / 60:.1f} min, eta {(total - step) * el / step / 60:.0f} min",
                  flush=True)
        if step in eval_at:
            te0 = time.time()
            sc = score_split(model, pbs["dev"], dv_rank, a, device)
            f1, _ = eval_dev(sc, dv_rank, gold_d)
            rec = {"step": step, "epoch": round(step / steps_epoch, 2), "ce_dev_f1": round(f1, 4),
                   "stage1_f1": round(base_f1, 4), "time_min": round((time.time() - t0) / 60, 1)}
            if f1 > best:
                best = f1
                rec["best"] = True
                save_scores(sc, "dev", a.out)
                save_scores(score_split(model, pbs["test"], te_rank, a, device), "test", a.out)
                json.dump(rec, open(os.path.join(a.out, "best.json"), "w"), indent=1)
            rec["eval_min"] = round((time.time() - te0) / 60, 1)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
    print(f"done in {(time.time() - t0) / 60:.0f} min; best dev CE F1@7 {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
