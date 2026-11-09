"""Masked task SFT: fine-tune a pretrained checkpoint on prompt -> 128-word continuation pairs, with the target
tokens of specifics NOT supported by the prompt optionally masked out of the loss, plus plain-text replay.

  CUDA_VISIBLE_DEVICES=0 python phase1/sft/train_sft.py --config phase1/configs/sft/archE_sft_unsupported.json
  ... --max-steps 20 --name smoke      # smoke test

New code only: phase1/train.py, the models, the tokenizer and generation are untouched. Outputs go to
runs/<name>/: config.json, mask_stats.json, plan_stats.json, metrics.jsonl, ckpt_best.pt, ckpt_final.pt (model-only,
same format as pretraining checkpoints, so phase1/generate.py loads them unchanged).

Data (built by phase1/sft/build_pairs.py and build_replay.py):
  pairs   data/sft/masked_sft_v1/{train,val}.pt: prompt ids, target ids, per-target-token unsupported/supported flags
  replay  data/sft/replay_v1/<source>.bin: cleaned pretraining text, documents joined by <|endoftext|>

Rows (row_len tokens) are packed with whole examples, each one its own document (followed by <|endoftext|>), so with
the model's doc_mask an example only attends to itself and its RoPE positions start at 0, exactly as at generation time.
  SFT example   [prompt][target][eos]   labels only on the target tokens; mask_mode decides which:
                  none                  all target tokens
                  unsupported           target tokens of specifics not in the prompt -> -100
                  unsupported+upweight  as above, supported-specific tokens weighted supported_weight
  replay chunk  a random window (<= 1536 tokens) of a replay stream + [eos]: plain LM labels (boundary labels masked
                as in pretraining doc_mask; the label of the artificial final eos is -100)
Replay fills the space left in the rows after packing, plus extra rows if needed, so that replay makes up
replay_ratio of the step's loss tokens (replay_basis "loss_tokens", default; SFT counted as its target tokens before
masking, so all three mask modes get the byte-identical plan) or of its INPUT tokens ("input_tokens").
Deficits carry over to the next step, so the ratio holds over the run.

Loss of a step = sum(weight * nll) over all loss tokens of the step / number of loss tokens of the step
(weights only change the numerator, so with supported_weight 1.0 "unsupported+upweight" == "unsupported").

Facts block (ceiling test, build_facts_pairs.py; both flags off by default):
  data.facts               put each pair's facts block before its prompt (labels -100 on block and prompt)
  data.plan_reserve_facts  pack rows as if the block were there (control run: same plan as the facts run)

Validation every eval_every steps on the 1k val pairs (all target tokens scored, no masking), reported as
  val_all      mean nll over all target tokens
  val_common   mean nll over target tokens NOT inside an unsupported specific (the tokens every mode trains on;
               the same token set for all runs, used to pick ckpt_best)
  val_unsup / val_sup   nll on unsupported / supported specific tokens
plus the pretraining validation loss (data/tokens val windows, boundary-masked) to watch forgetting.
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_DIR, "phase1"))
sys.path.insert(0, HERE)
from data import fixed_val_windows, mask_boundary_labels  # noqa: E402
from masking import MASK_MODES, target_labels  # noqa: E402
from models import build_model  # noqa: E402

P = lambda *a: os.path.join(PROJECT_DIR, *a)
EOS = 0
MAX_DOC = 1536            # RoPE table length: no single document (example / replay chunk) may be longer
SOURCES = ["supreme_court", "high_courts", "tribunals", "acts"]
CAT_PAD, CAT_TGT, CAT_SUP, CAT_UNSUP = 0, 1, 2, 3


# ----------------------------------------------------------------------------- segments

def pair_segment(r, mask_mode, supported_weight, score_all=False, facts=False):
    """-> (ids, labels, weights, cats) for one SFT example. labels[j] is the target of position j.
    score_all: label every target token (validation), ignoring mask_mode.
    facts: put the pair's facts block (r["facts_ids"], build_facts_pairs.py) before the prompt; it is never a target."""
    p = r["prompt_ids"].astype(np.int64)
    if facts:
        p = np.concatenate([r["facts_ids"].astype(np.int64), p])
    t = r["target_ids"].astype(np.int64)
    ids = np.concatenate([p, t, [EOS]])
    L, P_, T = len(ids), len(p), len(t)
    labels = np.full(L, -100, dtype=np.int64)
    weights = np.zeros(L, dtype=np.float32)
    cats = np.zeros(L, dtype=np.int8)
    tl, tw = target_labels(t, r["unsup"], r["sup"], "none" if score_all else mask_mode, supported_weight)
    labels[P_ - 1:P_ - 1 + T] = tl            # position P-1+k predicts target token k
    weights[P_ - 1:P_ - 1 + T] = np.where(tl != -100, tw, 0.0)
    c = np.where(r["unsup"], CAT_UNSUP, np.where(r["sup"], CAT_SUP, CAT_TGT))
    cats[P_ - 1:P_ - 1 + T] = c
    return ids, labels, weights, cats


def replay_segment(chunk):
    """chunk: token window (np.uint16) -> (ids, labels, weights) with an eos separator appended."""
    ids = np.concatenate([chunk.astype(np.int64), [EOS]])
    labels = np.full(len(ids), -100, dtype=np.int64)
    labels[:-2] = ids[1:-1]                    # the artificial final eos is not a target
    labels[:-1][ids[:-1] == EOS] = -100        # doc_mask: never predict across a document boundary
    weights = (labels != -100).astype(np.float32)
    return ids, labels, weights


# ----------------------------------------------------------------------------- plan

class Plan:
    """Deterministic list of steps; step = list of rows; row = list of ("p", pair_idx) / ("r", src, start, length)."""

    def __init__(self, pairs, replay, cfg, seed):
        s, t, d = cfg["sft"], cfg["train"], cfg["data"]
        self.facts = bool(d.get("facts", False))
        self.row_len, self.rows_per_step = t["row_len"], t["rows_per_step"]
        self.ratio, self.basis = s["replay_ratio"], s.get("replay_basis", "loss_tokens")
        if self.basis not in ("loss_tokens", "input_tokens"):
            raise ValueError("replay_basis must be loss_tokens or input_tokens")
        self.replay = replay
        mix = cfg["data"]["replay_mix"]
        self.rsrc = [k for k in SOURCES if mix.get(k, 0) > 0]
        self.rw = np.array([mix[k] for k in self.rsrc], dtype=float)
        self.rw /= self.rw.sum()
        # plan_reserve_facts: pack as if the facts block were present (also when it is not used), so a run with the
        # block and its control without it get the identical plan (rows, replay chunks, step order)
        reserve = self.facts or bool(d.get("plan_reserve_facts", False))
        self.seg_len = np.array([len(r["prompt_ids"]) + len(r["target_ids"]) + 1 +
                                 (len(r["facts_ids"]) if reserve else 0) for r in pairs])
        # target tokens BEFORE masking: the plan (rows, replay chunks) is then identical for every mask_mode
        self.seg_loss = np.array([len(r["target_ids"]) for r in pairs])
        rng = np.random.default_rng(seed)
        self.steps, self.stats = [], defaultdict(int)
        carry = 0.0
        for ep in range(t["epochs"]):
            queue = list(rng.permutation(len(pairs)))
            while queue:
                rows, free = [], []
                for _ in range(self.rows_per_step):
                    if not queue:
                        break
                    row, f = [], self.row_len
                    j = 0
                    while j < min(len(queue), 64):             # first fit within a 64-example lookahead
                        i = queue[j]
                        if self.seg_len[i] <= f:
                            row.append(("p", int(i)))
                            f -= self.seg_len[i]
                            queue.pop(j)
                        else:
                            j += 1
                    rows.append(row)
                    free.append(f)
                sft_loss = sum(self.seg_loss[x[1]] for row in rows for x in row)
                sft_in = sum(self.seg_len[x[1]] for row in rows for x in row)
                base = sft_loss if self.basis == "loss_tokens" else sft_in
                need = base * self.ratio / (1 - self.ratio) + carry
                # fill gaps, then whole extra rows, until the step's replay tokens reach `need`
                ri = 0
                while need > 1:
                    if ri == len(rows):
                        if need < self.row_len / 2 and len(rows) >= self.rows_per_step:
                            break                               # small remainder: carry it to the next step
                        rows.append([])
                        free.append(self.row_len)
                    while free[ri] >= 16 and need > 1:
                        g = int(min(free[ri], MAX_DOC + 1, max(16, need + 2)))
                        src = int(rng.choice(len(self.rsrc), p=self.rw))
                        stream = self.replay[self.rsrc[src]]
                        start = int(rng.integers(0, len(stream) - g))
                        _, lab, _ = replay_segment(stream[start:start + g - 1])
                        rows[ri].append(("r", src, start, g - 1))
                        free[ri] -= g
                        got = int((lab != -100).sum()) if self.basis == "loss_tokens" else g
                        need -= got
                        self.stats["replay_loss_tokens"] += int((lab != -100).sum())
                        self.stats["replay_input_tokens"] += g
                    ri += 1
                carry = max(0.0, need)
                self.stats["sft_loss_tokens"] += int(sft_loss)
                self.stats["sft_input_tokens"] += int(sft_in)
                self.stats["rows"] += len(rows)
                self.stats["pad_tokens"] += int(sum(free))
                self.steps.append(rows)
            self.stats[f"steps_epoch{ep + 1}"] = len(self.steps)
        n_in = self.stats["sft_input_tokens"] + self.stats["replay_input_tokens"]
        self.stats.update(
            steps=len(self.steps), total_row_tokens=self.stats["rows"] * self.row_len,
            replay_share_of_loss_tokens=self.stats["replay_loss_tokens"] / max(1, self.stats["replay_loss_tokens"] + self.stats["sft_loss_tokens"]),
            replay_share_of_input_tokens=self.stats["replay_input_tokens"] / max(1, n_in),
            pad_share=self.stats["pad_tokens"] / max(1, self.stats["rows"] * self.row_len))

    def materialize(self, rows, pairs, mask_mode, supported_weight):
        """rows -> x, y (int64, (R, row_len)), w (float32), is_replay (bool) tensors."""
        R = len(rows)
        x = np.full((R, self.row_len), EOS, dtype=np.int64)
        y = np.full((R, self.row_len), -100, dtype=np.int64)
        w = np.zeros((R, self.row_len), dtype=np.float32)
        rep = np.zeros((R, self.row_len), dtype=bool)
        for i, row in enumerate(rows):
            pos = 0
            for seg in row:
                if seg[0] == "p":
                    ids, lab, wt, _ = pair_segment(pairs[seg[1]], mask_mode, supported_weight,
                                                   facts=getattr(self, "facts", False))
                else:
                    _, src, start, n = seg
                    ids, lab, wt = replay_segment(self.replay[self.rsrc[src]][start:start + n])
                    rep[i, pos:pos + len(ids)] = True
                x[i, pos:pos + len(ids)], y[i, pos:pos + len(ids)], w[i, pos:pos + len(ids)] = ids, lab, wt
                pos += len(ids)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w), torch.from_numpy(rep)


def load_replay(path):
    return {s: np.memmap(os.path.join(path, f"{s}.bin"), dtype=np.uint16, mode="r")
            for s in SOURCES if os.path.exists(os.path.join(path, f"{s}.bin"))}


# ----------------------------------------------------------------------------- loss / eval

def step_loss(logits, y, w, n_loss_tokens):
    """sum(w * nll) / n_loss_tokens (the step's total number of loss tokens)."""
    nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="none")
    return (nll * w.reshape(-1)).sum() / n_loss_tokens, nll


def pack_val(pairs, row_len, facts=False):
    """Validation rows: every target token scored; cats/source per position for the breakdown."""
    rows, cur, f = [], [], row_len
    for i in range(len(pairs)):
        n = len(pairs[i]["prompt_ids"]) + len(pairs[i]["target_ids"]) + 1 + (len(pairs[i]["facts_ids"]) if facts else 0)
        if n > f:
            rows.append(cur)
            cur, f = [], row_len
        cur.append(i)
        f -= n
    rows.append(cur)
    X, Y, C, S = [], [], [], []
    for row in rows:
        x = np.full(row_len, EOS, dtype=np.int64); y = np.full(row_len, -100, dtype=np.int64)
        c = np.zeros(row_len, dtype=np.int8); s = np.full(row_len, -1, dtype=np.int8)
        pos = 0
        for i in row:
            ids, lab, _, cat = pair_segment(pairs[i], "none", 1.0, score_all=True, facts=facts)
            n = len(ids)
            x[pos:pos + n], y[pos:pos + n], c[pos:pos + n] = ids, lab, cat
            s[pos:pos + n] = SOURCES.index(pairs[i]["source"])
            pos += n
        X.append(x); Y.append(y); C.append(c); S.append(s)
    out = [torch.from_numpy(np.stack(a)) for a in (X, Y, C, S)]
    return out


def pad_rows(tensors, multiple, fills):
    """Pad the row dimension to a multiple (static shapes for torch.compile)."""
    n = -len(tensors[0]) % multiple
    if not n:
        return tensors
    return [torch.cat([t, torch.full((n, *t.shape[1:]), f, dtype=t.dtype)]) for t, f in zip(tensors, fills)]


@torch.no_grad()
def evaluate_sft(model, val, micro_rows):
    model.eval()
    X, Y, C, S = val
    tot = defaultdict(float)
    cnt = defaultdict(int)
    for i in range(0, len(X), micro_rows):
        x, y = X[i:i + micro_rows].cuda(), Y[i:i + micro_rows].cuda()
        c, s = C[i:i + micro_rows].cuda().reshape(-1), S[i:i + micro_rows].cuda().reshape(-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="none")
        groups = {"all": c > 0, "common": (c > 0) & (c != CAT_UNSUP), "unsup": c == CAT_UNSUP, "sup": c == CAT_SUP}
        for g, m in groups.items():
            tot[g] += nll[m].sum().item(); cnt[g] += int(m.sum())
            for si, src in enumerate(SOURCES):
                ms = m & (s == si)
                tot[f"{g}/{src}"] += nll[ms].sum().item(); cnt[f"{g}/{src}"] += int(ms.sum())
    model.train()
    return {k: tot[k] / cnt[k] for k in tot if cnt[k]}


@torch.no_grad()
def evaluate_pretrain(model, sets, micro):
    model.eval()
    out = {}
    for name, (x, y) in sets.items():
        y = mask_boundary_labels(x, y, EOS)
        tot, n = 0.0, 0
        for i in range(0, len(x), micro):
            xb, yb = x[i:i + micro].cuda(), y[i:i + micro].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(xb)
            nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), yb.reshape(-1), ignore_index=-100, reduction="sum")
            tot, n = tot + nll.item(), n + int((yb != -100).sum())
        out[name] = tot / n
    out["mean"] = sum(out.values()) / len(out)
    model.train()
    return out


def mask_stats(pairs, mask_mode, supported_weight):
    """Share of target tokens masked / up-weighted and specifics per target, per source (for this run's mode)."""
    out = {}
    by = defaultdict(list)
    for r in pairs:
        by[r["source"]].append(r)
        by["all"].append(r)
    for g, rs in by.items():
        n = sum(len(r["target_ids"]) for r in rs)
        masked = upw = 0
        for r in rs:
            lab, w = target_labels(r["target_ids"], r["unsup"], r["sup"], mask_mode, supported_weight)
            masked += int((lab == -100).sum())
            upw += int(((w != 1.0) & (lab != -100)).sum())
        out[g] = dict(pairs=len(rs), target_tokens=n, share_masked=masked / n, share_upweighted=upw / n,
                      specifics_per_target_supported=float(np.mean([r["n_spec_sup"] for r in rs])),
                      specifics_per_target_unsupported=float(np.mean([r["n_spec_unsup"] for r in rs])))
    return out


def lr_at(step, total, t):
    warm = max(1, math.ceil(t["warmup_frac"] * total))
    peak, floor = t["lr"], t["lr"] * t["min_lr_frac"]
    if step < warm:
        return peak * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    if t.get("schedule", "cosine") == "linear":
        return floor + (peak - floor) * (1 - prog)
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * prog))


def make_optimizer(model, t):
    import torch.nn as nn
    linear = {id(m.weight) for m in model.modules() if isinstance(m, nn.Linear)} - {id(model.embed.weight)}
    decay = [p for p in model.parameters() if id(p) in linear]
    no_decay = [p for p in model.parameters() if id(p) not in linear]
    return torch.optim.AdamW([{"params": decay, "weight_decay": t["weight_decay"]},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=t["lr"], betas=(0.9, t["beta2"]), eps=1e-8, fused=True)


def save(path, model, step, cfg, extra=None):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "step": step, "config": cfg, **(extra or {})}, tmp)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--plan-only", action="store_true", help="build the plan, print its stats and exit (no GPU)")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    if args.name:
        cfg["name"] = args.name
    s, t, d = cfg["sft"], cfg["train"], cfg["data"]
    if s["mask_mode"] not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of {MASK_MODES}")
    torch.manual_seed(t.get("seed", 0))
    torch.backends.cuda.matmul.allow_tf32 = True

    pairs = torch.load(P(d["pairs"], "train.pt"), weights_only=False)
    vpairs = torch.load(P(d["pairs"], "val.pt"), weights_only=False)
    if d.get("max_pairs"):
        pairs = pairs[:d["max_pairs"]]
    replay = load_replay(P(d["replay"]))
    t0 = time.time()
    plan = Plan(pairs, replay, cfg, seed=t.get("seed", 0))
    print(f"plan: {len(plan.steps)} steps, {dict(plan.stats)} ({time.time() - t0:.0f}s)", flush=True)
    if args.plan_only:
        return
    run_dir = P("runs", cfg["name"])
    os.makedirs(run_dir, exist_ok=True)

    init = torch.load(P(cfg["init_ckpt"]), map_location="cpu", weights_only=False)
    model_cfg = dict(init["config"]["model"])
    for k, v in cfg.get("model_overrides", {}).items():
        model_cfg[k] = v
    model, _ = build_model(model_cfg)
    model.load_state_dict(init["model"])
    model = model.cuda()
    if not model_cfg.get("doc_mask"):
        raise ValueError("packed SFT rows need model.doc_mask=true")
    out_cfg = {**init["config"], "model": model_cfg, "name": cfg["name"], "sft_config": cfg,
               "init_step": init["step"]}
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    ms = mask_stats(pairs, s["mask_mode"], s.get("supported_weight", 2.0))
    json.dump(ms, open(os.path.join(run_dir, "mask_stats.json"), "w"), indent=1)
    json.dump(dict(plan.stats), open(os.path.join(run_dir, "plan_stats.json"), "w"), indent=1)
    print("mask stats (all):", json.dumps(ms["all"]), flush=True)
    for src in SOURCES:
        print(f"  {src}: masked {ms[src]['share_masked']:.4f} upweighted {ms[src]['share_upweighted']:.4f} "
              f"spec/target sup {ms[src]['specifics_per_target_supported']:.2f} unsup {ms[src]['specifics_per_target_unsupported']:.2f}")

    opt = make_optimizer(model, t)
    # a few fixed shapes (train rows, val rows, 1536-token pretraining-val windows; grad and no-grad): compile each one
    # statically and keep them all (the default limit of 8 made a frame fall back to slow eager flex_attention)
    from torch import _dynamo
    _dynamo.config.cache_size_limit = 64
    _dynamo.config.automatic_dynamic_shapes = False
    train_model = torch.compile(model) if t.get("compile", True) else model
    val = pad_rows(pack_val(vpairs, t["row_len"], facts=plan.facts), t["micro_rows"], [EOS, -100, CAT_PAD, -1])
    pre_sets = {n: pad_rows(list(fixed_val_windows(d.get("tokenizer", "legal_bpe_32k"), n, 1536,
                                                   t.get("pretrain_val_windows", 32))), t["micro_rows"], [EOS, -100])
                for n in SOURCES}
    log = open(os.path.join(run_dir, "metrics.jsonl"), "a")
    total = len(plan.steps) if not args.max_steps else min(len(plan.steps), args.max_steps)
    micro = t["micro_rows"]
    sw = s.get("supported_weight", 2.0)
    best = float("inf")

    def do_eval(step):
        nonlocal best
        v = evaluate_sft(train_model, val, micro)          # compiled: eager flex_attention is very slow
        pv = evaluate_pretrain(train_model, pre_sets, micro)
        rec = {"step": step, "val": v, "pretrain_val": pv}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"  [eval step {step}] val_all {v['all']:.4f} val_common {v['common']:.4f} unsup {v['unsup']:.4f} "
              f"sup {v['sup']:.4f} | pretrain val {pv['mean']:.4f}", flush=True)
        if v["common"] < best:
            best = v["common"]
            save(os.path.join(run_dir, "ckpt_best.pt"), model, init["step"] + step, out_cfg, {"val": v, "pretrain_val": pv})
            print(f"  new best val_common {best:.4f} -> ckpt_best.pt", flush=True)

    do_eval(0)
    tin, tstart = 0, time.time()
    for step in range(total):
        lr = lr_at(step, total, t)
        for g in opt.param_groups:
            g["lr"] = lr
        rows = plan.steps[step]
        rows = rows + [[]] * (-len(rows) % micro)          # empty rows (no loss tokens): every micro-batch has micro rows
        x, y, w, rep = plan.materialize(rows, pairs, s["mask_mode"], sw)
        n_loss = int((y != -100).sum())
        sums = {"sft": [0.0, 0], "replay": [0.0, 0]}
        for i in range(0, len(x), micro):
            xb, yb, wb, rb = (a[i:i + micro].cuda(non_blocking=True) for a in (x, y, w, rep))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = train_model(xb)
            loss, nll = step_loss(logits, yb, wb, n_loss)
            loss.backward()
            with torch.no_grad():
                valid = (yb != -100).reshape(-1)
                r = rb.reshape(-1)
                sums["sft"][0] += nll[valid & ~r].sum().item(); sums["sft"][1] += int((valid & ~r).sum())
                sums["replay"][0] += nll[valid & r].sum().item(); sums["replay"][1] += int((valid & r).sum())
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"]).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        tin += x.numel()
        if step % t.get("log_every", 10) == 0 or step == total - 1:
            el = time.time() - tstart
            rec = {"step": step, "lr": lr, "grad_norm": gn, "loss_sft": sums["sft"][0] / max(1, sums["sft"][1]),
                   "loss_replay": sums["replay"][0] / max(1, sums["replay"][1]), "sft_tokens": sums["sft"][1],
                   "replay_tokens": sums["replay"][1], "rows": len(x), "tok_per_s": tin / el,
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"step {step}/{total} sft {rec['loss_sft']:.4f} replay {rec['loss_replay']:.4f} lr {lr:.2e} gn {gn:.2f} "
                  f"{rec['tok_per_s'] / 1e3:.0f}k tok/s mem {rec['peak_mem_gb']:.1f}GB eta {(total - step - 1) * el / (step + 1) / 60:.0f}m",
                  flush=True)
        if (step + 1) % t["eval_every"] == 0 and step + 1 < total:
            do_eval(step + 1)
        if t.get("snapshot_every") and (step + 1) % t["snapshot_every"] == 0 and step + 1 < total:
            save(os.path.join(run_dir, f"ckpt_step{step + 1}.pt"), model, init["step"] + step + 1, out_cfg)
    do_eval(total)
    save(os.path.join(run_dir, "ckpt_final.pt"), model, init["step"] + total, out_cfg)
    print("done", flush=True)


if __name__ == "__main__":
    main()
