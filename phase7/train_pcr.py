"""Dense bi-encoder for Prior Case Retrieval on a Phase-1 decoder (Arch E / F), contrastive fine-tuning.

Document -> `spread` chunks (max_chunks x max_len tokens spread evenly over the case, phase4 doc_chunks), packed into
rows with FlexAttention segment masks (bidirectional within a chunk, RoPE restarts per chunk; phase4 cjpe_model).
Chunk vector = mean of its token states; document vector = mean (or attention pool) of its chunk vectors, L2-normalised.
Score = cosine / tau.

Training (one step = --bq queries): for each query one gold candidate sampled at random + --n_hard hard negatives from
the query's BM25 top --hard_pool (bigram BM25, not gold, not the query) + in-batch negatives (every other document of
the step). InfoNCE over all step documents; the query's other gold candidates and the query itself are masked out.
Gradient caching (Gao et al. 2021, GradCache): embeddings of all step documents are computed without a graph, the loss
gradient w.r.t. the embeddings is taken, then every micro-batch is re-encoded with a graph and back-propagated with its
cached gradient -> large contrastive batches at the memory of one micro-batch. Dropout is 0 so both passes agree.

Techniques: full FT, --peft lorma|lora|ia3|prompt (prompt tuning: --n_prompt learned input vectors prepended to every
chunk, backbone frozen), --train_layers k (partial FT: top k blocks), --n_layers (student) + --kd teacher score file
(listwise KD: KL between teacher and student softmax over the step documents, at temperature --kd_T).

Evaluation: dev F1@7 of the dense score alone and of z-score fusion with bigram BM25 (z(bm25) + w z(dense)).
Outputs in --out: config.json, log.jsonl, best.json, emb_<split>.npz (doc ids, doc vectors, chunk vectors) at the
best evaluation, adapter.pt (--save).
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

sys.path.insert(0, os.path.join(D.ROOT, "phase4"))
sys.path.insert(0, os.path.join(D.ROOT, "phase2"))
from cjpe_data import doc_chunks, micro_batches  # noqa: E402
from cjpe_model import load_backbone, segment_masks  # noqa: E402
from peft import add_adapters  # noqa: E402

PAD_SEG = 1 << 20
LEX = os.path.join(D.ROOT, "runs_p7", "lex")


class BiEncoder(nn.Module):
    def __init__(self, backbone, doc_pool="mean", n_prompt=0):
        super().__init__()
        self.backbone = backbone
        backbone.lm_head = None
        for block in backbone.blocks:
            block.drop.p = 0.0
        self.window = backbone.cfg.window
        d = backbone.cfg.d_model
        self.doc_pool = doc_pool
        if doc_pool == "attn":
            self.att = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
            nn.init.zeros_(self.att[2].weight)
            nn.init.zeros_(self.att[2].bias)
        self.n_prompt = n_prompt
        if n_prompt:   # prompt tuning (Lester et al. 2021): init from the embeddings of random vocabulary tokens
            idx = torch.randint(100, backbone.embed.weight.size(0) - 100, (n_prompt,))
            self.prompt = nn.Parameter(backbone.embed.weight.detach()[idx].clone())
        self.compiled = None

    def set_compiled(self, fn):
        object.__setattr__(self, "compiled", fn)

    def encoder(self, ids, pos, pidx, masks):
        bb = self.backbone
        x = bb.embed(ids)
        if self.n_prompt:
            x = torch.where((pidx >= 0)[..., None], self.prompt.to(x.dtype)[pidx.clamp(min=0)], x)
        cos, sin = bb.rope_cos[pos], bb.rope_sin[pos]
        for block in bb.blocks:
            x = block(x, cos, sin, masks[block.attn.local])
        return bb.norm(x)

    def forward(self, b):
        """-> (doc vectors (n_docs, d) L2-normalised fp32, chunk vectors (n_chunks, d) normalised, chunk->doc)."""
        masks = segment_masks(b["seg"], self.window)
        fn = self.encoder if self.compiled is None else self.compiled
        h = fn(b["ids"], b["pos"], b["pidx"], masks).float()
        d = h.size(-1)
        tok = b["tok_chunk"].view(-1)
        keep = tok >= 0
        n_chunks = b["chunk_doc"].numel()
        csum = h.new_zeros(n_chunks, d).index_add_(0, tok[keep], h.view(-1, d)[keep])
        cnt = h.new_zeros(n_chunks).index_add_(0, tok[keep], h.new_ones(int(keep.sum())))
        c = csum / cnt.clamp(min=1)[:, None]
        slots = b["doc_slots"]
        ok = slots >= 0
        cv = c[slots.clamp(min=0)]
        if self.doc_pool == "attn":
            w = self.att(cv).squeeze(-1).masked_fill(~ok, float("-inf")).softmax(-1)
        else:
            w = ok.float() / ok.sum(-1, keepdim=True)
        v = (w[..., None] * cv).sum(1)
        return F.normalize(v, dim=-1), F.normalize(c, dim=-1), b["chunk_doc"]


def pack(mb, chunks, rows, seq, n_prompt, device):
    ids = torch.zeros(rows, seq, dtype=torch.long)
    pos = torch.zeros(rows, seq, dtype=torch.long)
    pidx = torch.full((rows, seq), -1, dtype=torch.long)
    seg = torch.full((rows, seq), PAD_SEG, dtype=torch.long)
    tok_chunk = torch.full((rows, seq), -1, dtype=torch.long)
    K = max(len(p) for _, p in mb)
    doc_slots = torch.full((len(mb), K), -1, dtype=torch.long)
    chunk_doc, ci = [], 0
    P = n_prompt
    for j, (d, place) in enumerate(mb):
        for k, ((c, _), (r, o)) in enumerate(zip(chunks[d], place)):
            L = len(c)                                  # includes the P prompt slots (chunks are built with them)
            ids[r, o + P:o + L] = torch.from_numpy(c[P:])
            pos[r, o:o + L] = torch.arange(L)
            if P:
                pidx[r, o:o + P] = torch.arange(P)
            seg[r, o:o + L] = ci
            tok_chunk[r, o + P:o + L] = ci
            doc_slots[j, k] = ci
            chunk_doc.append(j)
            ci += 1
    t = lambda x: x.to(device, non_blocking=True)  # noqa: E731
    return {"ids": t(ids), "pos": t(pos), "pidx": t(pidx), "seg": t(seg), "tok_chunk": t(tok_chunk),
            "doc_slots": t(doc_slots), "chunk_doc": t(torch.tensor(chunk_doc)), "docs": [d for d, _ in mb]}


def make_chunks(toks, a):
    out = []
    for t in toks:
        ch = doc_chunks(t, "spread", a.max_len - a.n_prompt, a.max_chunks)
        if a.n_prompt:
            ch = [(np.concatenate([np.zeros(a.n_prompt, dtype=t.dtype), c]), s) for c, s in ch]
        out.append(ch)
    return out


def encode(model, chunks, idx, a, device, grad=False):
    """Encode documents idx (list of indices into chunks) -> doc vectors (len(idx), d) in idx order."""
    order = sorted(range(len(idx)), key=lambda i: -sum(len(c) for c, _ in chunks[idx[i]]))
    sub = [chunks[i] for i in idx]
    out_v = [None] * len(idx)
    out_c = [None] * len(idx)
    for mb in micro_batches(sub, order, a.rows, a.seq):
        b = pack(mb, sub, a.rows, a.seq, a.n_prompt, device)
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
            v, c, cd = model(b)
        for j, d in enumerate(b["docs"]):
            out_v[d] = v[j]
            out_c[d] = c[cd == j]
    return torch.stack(out_v), out_c


@torch.no_grad()
def embed_split(model, chunks, a, device):
    model.eval()
    v, c = encode(model, chunks, list(range(len(chunks))), a, device)
    model.train()
    return v.float().cpu().numpy(), [x.float().cpu().numpy() for x in c]


def zrow(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


def evaluate(split_data, V, qrow, ws=(0.0, 0.25, 0.5, 1.0, 2.0)):
    qids, cids, gold, bm, pos = split_data
    qi = [qrow[q] for q in qids]
    S = V[qi] @ V[:len(cids)].T                             # (n_q, n_pool) cosine, pool order = cids
    rec = {"dense": D.score(gold, D.rank_from_scores(qids, cids, S))["f1"]}
    zb, zd = zrow(bm), zrow(S)
    for w in ws:
        rec[f"fuse{w}"] = D.score(gold, D.rank_from_scores(qids, cids, zb + w * zd))["f1"]
    return rec, S


def load_split(split, bm_name):
    ids, toks = load_tokens(split)
    qs, gold, pool = D.load(split)
    assert ids == list(pool)
    z = np.load(os.path.join(LEX, f"{bm_name}_{split}.npz"))
    assert list(z["cids"]) == ids
    bm = z["S"].astype(np.float64)
    qids = list(z["qids"])
    pos = {c: j for j, c in enumerate(ids)}
    return ids, toks, (qids, ids, gold, bm, pos)


def query_ctx_tokens(split):
    """Token arrays of the <CITATION>-context view of every query (features.ctx_text), cached."""
    path = os.path.join(D.ROOT, "runs_p7", "cache", f"{split}_qctx.npz")
    if not os.path.exists(path):
        from tokenizers import Tokenizer
        from features import ctx_text
        from pcr_tok import TOKENIZER
        q = json.load(open(os.path.join(D.DATA, D.FILES[split][0])))
        ids = list(q)
        enc = Tokenizer.from_file(TOKENIZER).encode_batch([ctx_text(q[k]["text"]) for k in ids])
        np.savez(path, ids=np.array(ids), lens=np.array([len(e.ids) for e in enc]),
                 flat=np.concatenate([np.asarray(e.ids, dtype=np.uint16) for e in enc]))
    z = np.load(path)
    offs = np.concatenate([[0], np.cumsum(z["lens"])])
    flat = z["flat"].astype(np.int32)
    return {str(d): flat[offs[i]:offs[i + 1]] for i, d in enumerate(z["ids"])}


def split_chunks(split, ids, toks, a):
    """-> chunk list (pool docs first, then the query-view docs if --query_view ctx), {qid: chunk-list row}."""
    ch = make_chunks([toks[i] for i in ids], a)
    pos = {c: j for j, c in enumerate(ids)}
    qids = list(json.load(open(os.path.join(D.DATA, D.FILES[split][0]))))
    if a.query_view == "full":
        return ch, {q: pos[q] for q in qids}
    qt = query_ctx_tokens(split)
    ch = ch + make_chunks([qt[q] for q in qids], a)
    return ch, {q: len(ids) + k for k, q in enumerate(qids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(D.ROOT, "runs/sft_archE_v2_upweight/ckpt_best.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--max_chunks", type=int, default=8)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--bq", type=int, default=16, help="queries per step")
    ap.add_argument("--n_hard", type=int, default=3)
    ap.add_argument("--hard_pool", type=int, default=30)
    ap.add_argument("--bm", default="bm25bi", help="lexical score file for hard negatives and fusion")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=8)
    ap.add_argument("--evals_per_epoch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--doc_pool", default="mean", choices=["mean", "attn"])
    ap.add_argument("--peft", default="none", choices=["none", "lorma", "lora", "rslora", "ia3", "bitfit", "prompt"])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--n_prompt", type=int, default=0)
    ap.add_argument("--train_layers", type=int, default=None)
    ap.add_argument("--n_layers", type=int, default=None)
    ap.add_argument("--kd", default=None, help="teacher run dir with train_scores.npz (qids, cids, S)")
    ap.add_argument("--kd_T", type=float, default=1.0, help="teacher softmax temperature on its own logits (S/tau_t)")
    ap.add_argument("--kd_alpha", type=float, default=0.5, help="weight of the gold InfoNCE (1 - alpha on KL)")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--inference", action="store_true", help="no training: load <out>/adapter.pt, embed dev + test")
    ap.add_argument("--train_scores", action="store_true", help="save the train-split score matrix (KD teacher)")
    ap.add_argument("--max_steps", type=int, default=None, help="smoke")
    ap.add_argument("--query_view", default="full", choices=["full", "ctx"],
                    help="ctx: encode queries from their <CITATION>-context sentences (candidates: full text)")
    a = ap.parse_args()
    if a.peft == "prompt":
        a.n_prompt = a.n_prompt or 20
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device(f"cuda:{a.gpu}")
    torch.cuda.set_device(device)

    tr_ids, tr_toks, tr = load_split("train", a.bm)
    dv_ids, dv_toks, dv = load_split("dev", a.bm)
    te_ids, te_toks = load_tokens("test")
    tr_ch, tr_qrow = split_chunks("train", tr_ids, tr_toks, a)
    dv_ch, dv_qrow = split_chunks("dev", dv_ids, dv_toks, a)
    te_ch, te_qrow = split_chunks("test", te_ids, te_toks, a)
    qids, _, gold, bm, pos = tr
    hard = {}
    for i, q in enumerate(qids):
        g = set(gold[q]) | {q}
        order = np.argsort(-bm[i])
        hard[q] = [tr_ids[j] for j in order[:a.hard_pool + len(g)] if tr_ids[j] not in g][:a.hard_pool]
    train_q = [q for q in qids if gold[q]]
    teacher = None
    if a.kd:
        z = np.load(os.path.join(a.kd, "train_scores.npz"))
        assert list(z["cids"]) == tr_ids
        tq = {q: i for i, q in enumerate(z["qids"])}
        teacher = (tq, torch.from_numpy(z["S"].astype(np.float32)), float(json.load(open(os.path.join(a.kd, "config.json")))["tau"]))

    backbone, cfg = load_backbone(a.ckpt, max(4096, a.max_len))
    if a.n_layers:
        backbone.blocks = backbone.blocks[:a.n_layers]
        print(f"student: first {a.n_layers} of {cfg.n_layers} blocks", flush=True)
    model = BiEncoder(backbone, a.doc_pool, a.n_prompt)
    if a.peft == "prompt":
        for p in model.backbone.parameters():
            p.requires_grad = False
    elif a.peft != "none":
        n = add_adapters(model.backbone, a.peft, a.rank, a.alpha)
        print(f"{a.peft}: {n} layers adapted", flush=True)
    elif a.train_layers is not None:
        nb = len(model.backbone.blocks)
        for name, p in model.backbone.named_parameters():
            top = name.startswith("blocks.") and int(name.split(".")[1]) >= nb - a.train_layers
            p.requires_grad = top or name.startswith("norm.")
        print(f"partial FT: top {a.train_layers} of {nb} blocks", flush=True)
    model = model.to(device)
    if not a.no_compile:
        from torch import _dynamo
        _dynamo.config.cache_size_limit = 64
        _dynamo.config.automatic_dynamic_shapes = False
        model.set_compiled(torch.compile(model.encoder, dynamic=False))
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    bb_params = {id(p) for p in model.backbone.parameters()}
    head = [p for p in model.parameters() if p.requires_grad and id(p) not in bb_params]
    decay, no_decay = [], []
    for name, p in model.backbone.named_parameters():
        if p.requires_grad:
            (no_decay if p.dim() < 2 or "norm" in name or "lambda" in name or "subln" in name else decay).append(p)
    groups = [{"params": decay, "lr": a.lr, "weight_decay": a.weight_decay},
              {"params": no_decay, "lr": a.lr, "weight_decay": 0.0}]
    if head:
        groups.append({"params": head, "lr": a.head_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, 0.98), eps=1e-6)
    base_lrs = [g["lr"] for g in opt.param_groups]

    steps_epoch = math.ceil(len(train_q) / a.bq)
    total = a.max_steps or int(steps_epoch * a.epochs)
    warm = max(1, int(a.warmup_frac * total))
    eval_at = {min(total, int(round(steps_epoch * k / a.evals_per_epoch)))
               for k in range(1, int(math.ceil(a.epochs * a.evals_per_epoch)) + 1)} | {total}
    print(f"params {n_params / 1e6:.2f}M trainable {n_train / 1e6:.3f}M | train queries {len(train_q)}, "
          f"{steps_epoch} steps/epoch, {total} total | spread {a.max_chunks}x{a.max_len} | bq {a.bq} hard {a.n_hard}",
          flush=True)
    tr_index = {c: j for j, c in enumerate(tr_ids)}

    def run_eval(step):
        V, C = embed_split(model, dv_ch, a, device)
        rec, S = evaluate(dv, V, dv_qrow)
        return rec, V, C, S

    def save_embeddings(V, C, Vt, Ct):
        for nm, ids_, V_, C_, qr in (("dev", dv_ids, V, C, dv_qrow), ("test", te_ids, Vt, Ct, te_qrow)):
            qk = list(qr)
            np.savez(os.path.join(a.out, f"emb_{nm}.npz"), ids=ids_, V=V_[:len(ids_)],
                     C=np.concatenate(C_[:len(ids_)]), nc=[len(x) for x in C_[:len(ids_)]],
                     qids=qk, QV=V_[[qr[q] for q in qk]],
                     QC=np.concatenate([C_[qr[q]] for q in qk]), qnc=[len(C_[qr[q]]) for q in qk])

    if a.inference:
        model.load_state_dict(torch.load(os.path.join(a.out, "adapter.pt"), map_location="cpu"), strict=False)
        rec, V, C, _ = run_eval(0)
        print(json.dumps({k: round(v, 4) for k, v in rec.items()}), flush=True)
        Vt, Ct = embed_split(model, te_ch, a, device)
        save_embeddings(V, C, Vt, Ct)
        return

    log = open(os.path.join(a.out, "log.jsonl"), "a")
    rec0, *_ = run_eval(0)
    print(json.dumps({"step": 0, **{k: round(v, 4) for k, v in rec0.items()}}), flush=True)
    log.write(json.dumps({"step": 0, **rec0}) + "\n")
    best, step, t0 = -1.0, 0, time.time()
    model.train()
    order = []
    while step < total:
        if len(order) < a.bq:
            order += random.sample(train_q, len(train_q))
        batch_q, order = order[:a.bq], order[a.bq:]
        # items = rows of tr_ch: pool documents (< n_pool, the candidates) and, with --query_view ctx, query views
        idx, slot = [], {}

        def add(j):
            if j not in slot:
                slot[j] = len(idx)
                idx.append(j)
            return slot[j]
        qs_i, pos_c = [], []
        for q in batch_q:
            qs_i.append(add(tr_qrow[q]))
            pos_c.append(tr_index[random.choice(gold[q])])
            add(pos_c[-1])
        for q in batch_q:
            for h in random.sample(hard[q], min(a.n_hard, len(hard[q]))):
                add(tr_index[h])
        cand = [j for j in idx if j < len(tr_ids)]                 # columns: candidate (pool) items only
        col = {j: k for k, j in enumerate(cand)}
        cand_s = [slot[j] for j in cand]
        # mask: for query row r, exclude its own doc and its other gold docs (keep the sampled positive)
        mask = torch.zeros(len(batch_q), len(cand), dtype=torch.bool)
        for r, q in enumerate(batch_q):
            for g in set(gold[q]) | {q}:
                j = tr_index[g]
                if j in col and j != pos_c[r]:
                    mask[r, col[j]] = True
        mask = mask.to(device)
        tgt = torch.tensor([col[j] for j in pos_c], device=device)
        docs = idx

        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * (step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
        with torch.no_grad():
            E, _ = encode(model, tr_ch, idx, a, device)
        E = E.float().detach().requires_grad_(True)
        logits = (E[qs_i] @ E[cand_s].T) / a.tau
        logits = logits.masked_fill(mask, -1e4)
        loss = F.cross_entropy(logits, tgt)
        if teacher is not None:
            tq, TS, ttau = teacher
            tl = torch.stack([TS[tq[q]][cand] for q in batch_q]).to(device) / ttau / a.kd_T
            tl = tl.masked_fill(mask, -1e4)
            pt = tl.softmax(-1)
            ls = (logits / a.kd_T).log_softmax(-1)
            kl = (pt * (pt.clamp_min(1e-12).log() - ls)).sum(-1).mean()
            loss = a.kd_alpha * loss + (1 - a.kd_alpha) * a.kd_T ** 2 * kl
        loss.backward()
        G = E.grad.detach()
        # second pass with a graph, one micro-batch at a time
        sub = [tr_ch[i] for i in idx]
        sorder = sorted(range(len(idx)), key=lambda i: -sum(len(c) for c, _ in sub[i]))
        for mb in micro_batches(sub, sorder, a.rows, a.seq):
            b = pack(mb, sub, a.rows, a.seq, a.n_prompt, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v, _, _ = model(b)
            (v.float() * G[b["docs"]]).sum().backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0:
            el = time.time() - t0
            print(f"step {step}/{total} loss {loss.item():.4f} docs {len(docs)} {el / 60:.1f} min, "
                  f"eta {(total - step) * el / step / 60:.0f} min", flush=True)
        if step in eval_at:
            te0 = time.time()
            rec, V, C, S = run_eval(step)
            rec = {"step": step, "epoch": round(step / steps_epoch, 2), "loss": round(loss.item(), 4), **rec,
                   "time_min": round((time.time() - t0) / 60, 1),
                   "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 1)}
            sel = max(v for k, v in rec.items() if k.startswith("fuse") and k != "fuse0.0")
            if sel > best:
                best = sel
                rec["best"] = True
                Vt, Ct = embed_split(model, te_ch, a, device)
                save_embeddings(V, C, Vt, Ct)
                json.dump(rec, open(os.path.join(a.out, "best.json"), "w"), indent=1)
                if a.save or a.train_scores:
                    keep = {n for n, p in model.named_parameters() if p.requires_grad}
                    torch.save({k: v for k, v in model.state_dict().items() if k in keep},
                               os.path.join(a.out, "adapter.pt"))
            rec["eval_min"] = round((time.time() - te0) / 60, 1)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
    if a.train_scores:
        model.load_state_dict(torch.load(os.path.join(a.out, "adapter.pt")), strict=False)
        V, C = embed_split(model, tr_ch, a, device)
        qi = [tr_qrow[q] for q in qids]
        n = len(tr_ids)
        np.savez(os.path.join(a.out, "train_scores.npz"), qids=qids, cids=tr_ids, S=(V[qi] @ V[:n].T).astype(np.float32))
        np.savez(os.path.join(a.out, "emb_train.npz"), ids=tr_ids, V=V[:n], C=np.concatenate(C[:n]),
                 nc=[len(x) for x in C[:n]], qids=qids, QV=V[qi])
    print(f"done in {(time.time() - t0) / 60:.0f} min; best dev fused F1@7 {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
