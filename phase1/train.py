"""Phase-1 pretraining for the decoder-family models (Arch A, C, D).

Usage:
  python phase1/train.py --config phase1/configs/archD.json
  python phase1/train.py --config phase1/configs/archD.json --resume      # continue after a crash
  python phase1/train.py --config phase1/configs/archA.json --decay-from runs/archA/ckpt_step20000.pt
      # warmup-stable-decay: branch off a stable-phase snapshot and run only the decay
  python phase1/train.py --config phase1/configs/archD_4B.json --continue-from runs/archD/ckpt_step3001.pt
      # extend a run: model-only snapshot + fresh optimizer, lr re-warmed over --rewarm-steps

Everything a run does is determined by its JSON config; outputs go to runs/<name>/:
  config.json, metrics.jsonl (train + validation), ckpt_latest.pt (resumable),
  ckpt_step<N>.pt (model-only snapshots for checkpoint comparison),
  ckpt_best.pt (model-only, lowest mean validation loss so far).
Early stopping: set train.early_stop_patience (evaluations without improvement) and
train.early_stop_min_delta in the config; patience 0 or absent disables it.

Intra-document masking: model.doc_mask=true (Arch D only) restricts attention to the current
document, restarts RoPE positions per document, and ignores the '<|endoftext|> -> next document'
labels; micro-batches are then weighted by their number of valid labels.
train.val_boundary_masked=true also logs val_bm: validation loss with those boundary labels
ignored in every run, so masked and unmasked runs can be compared (and it selects ckpt_best).

Optional train keys (all off by default, so older configs train exactly as before):
  "wd_linear_only": true     weight decay only on nn.Linear weights (not the (tied) embedding,
                             norms, lambda vectors or per-head norm weights)
  "prefetch": 2              build the micro-batches of the next steps in a background thread
  "single_doc_eval": {"steps": [1001, 2001], "n_docs": 64, "reference": "runs/.../single_doc_val.json"}
                             after these step counts, validation loss on single-document rows
                             printed next to the reference
                             model's loss at the same step (same batch size -> same tokens)
Every log line also reports an MFU estimate, peak memory and, for Arch E, lambda per layer.
LR sweeps: --lr <peak> --name <run name> --max-steps <n> override the config.

Multi-GPU (DistributedDataParallel), launched with torchrun:
  torchrun --standalone --nproc_per_node 2 phase1/train.py --config phase1/configs/archF.json
Each step's accum micro-batches are split across the ranks (rank r takes micro r, r+W, ...), so a
run sees exactly the same data as on one GPU; accum must be divisible by the number of GPUs. The
doc_mask label-count weights use the total over all ranks. Rank 0 alone logs, evaluates and saves.
Every rank holds the full model and optimizer state and the data depends only on the step, so a run
can be resumed with --resume on a different number of GPUs (e.g. started on 1 GPU, continued on 2).

Graceful stop: `touch runs/<name>/STOP` -> the run finishes the current step, writes ckpt_latest.pt,
deletes STOP and exits (continue later with --resume). phase1/launch_archF.sh wraps this.

Continued training from the config (instead of --continue-from):
  "train": {"continue_from": "runs/.../ckpt_best.pt", "continue_tokens": 5e8,
            "lr": <peak>, "warmup_steps": <ramp>, "decay_frac": 1.0, ...}
  loads the model-only snapshot with a fresh optimizer and runs continue_tokens more tokens on its
  own warmup-stable-decay schedule (step counting resumes from the snapshot's step).
"""

import argparse
import builtins
import contextlib
import json
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from data import (MixedWindowSampler, Prefetcher, fixed_val_windows, mask_boundary_labels, mask_for_mlm,
                  single_doc_val_windows)
from models import build_model, config_dict

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")
N_SPECIAL = 105  # ids 0..104 are special tokens in legal_bpe_32k
PEAK_BF16_FLOPS = {"NVIDIA RTX A6000": 154.8e12}   # dense tensor-core peak, for the MFU estimate


def lr_at(step, t):
    """Warmup-stable-decay: linear warmup, constant peak, then linear decay over the last decay_frac."""
    total, warm = t["steps"], t["warmup_steps"]
    decay_start = max(warm, int(total * (1 - t["decay_frac"])))
    peak, floor = t["lr"], t["lr"] * t["min_lr_frac"]
    if step < warm:
        return peak * (step + 1) / warm
    if step < decay_start:
        return peak
    progress = (step - decay_start) / max(1, total - decay_start)
    return floor + (peak - floor) * (1 - progress)


def make_optimizer(model, t):
    """named_parameters() lists a tied matrix once, so it lands in exactly one group."""
    if t.get("wd_linear_only", False):
        linear = {id(m.weight) for m in model.modules() if isinstance(m, nn.Linear)}
        linear -= {id(model.embed.weight)}                          # tied lm_head = embedding: no decay
        decay = [p for n, p in model.named_parameters() if id(p) in linear]
        no_decay = [p for n, p in model.named_parameters() if id(p) not in linear]
    else:
        decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
        no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": t["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=t["lr"], betas=(0.9, t["beta2"]), eps=1e-8, fused=True)


def flops_per_token(model, seq_len):
    """Training FLOPs per token (forward + backward = 3x forward): 6 per matmul weight (lm_head
    included) plus the attention score / value matmuls over the average number of visible keys
    (causal, or capped by the window; document boundaries ignored). Differential layers have
    value heads twice as wide, which doubles the value matmul."""
    matmul = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear))
    pos = torch.arange(1, seq_len + 1).float()
    attn = 0.0
    for block in model.blocks:
        a = block.attn
        ctx = (pos.clamp(max=a.window) if getattr(a, "local", False) else pos).mean().item()
        width = a.q_proj.out_features
        attn += 2 * ctx * width * (3 if getattr(a, "diff", False) else 2)   # QK^T + AV (AV 2x wide if diff)
    return 6 * matmul + 3 * attn


@torch.no_grad()
def evaluate_single_doc(model, sets, micro_batch):
    """Per-source loss over the valid labels of single-document rows (see data.single_doc_val_windows)."""
    model.eval()
    out = {}
    for name, (x, y) in sets.items():
        tot, n = 0.0, 0
        for i in range(0, len(x), micro_batch):
            xb, yb = x[i:i + micro_batch].cuda(non_blocking=True), y[i:i + micro_batch].cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(xb, yb)
            k = int((yb != -100).sum())
            tot, n = tot + loss.item() * k, n + k
        out[name] = tot / n
    out["mean"] = sum(out.values()) / len(out)
    model.train()
    return out


@torch.no_grad()
def evaluate(model, val_sets, micro_batch, boundary_eos=None):
    """Mean validation loss per source. boundary_eos=<id>: labels after <|endoftext|> are ignored
    and batches are weighted by their number of valid labels."""
    model.eval()
    out = {}
    for name, (x, y) in val_sets.items():
        if boundary_eos is not None:
            y = mask_boundary_labels(x, y, boundary_eos)
        losses, weights = [], []
        for i in range(0, len(x), micro_batch):
            xb, yb = x[i:i + micro_batch].cuda(non_blocking=True), y[i:i + micro_batch].cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(xb, yb)
            losses.append(loss.item())
            weights.append(int((yb != -100).sum()) if boundary_eos is not None else 1)
        out[name] = sum(l * w for l, w in zip(losses, weights)) / sum(weights)
    out["mean"] = sum(out.values()) / len(out)
    model.train()
    return out


def save(path, model, optimizer, step, cfg, full=True, extra=None):
    state = {"model": model.state_dict(), "step": step, "config": cfg, **(extra or {})}
    if full:
        state["optimizer"] = optimizer.state_dict()
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--decay-from", default=None, help="snapshot to branch a decay-only run from")
    ap.add_argument("--continue-from", default=None,
                    help="model-only snapshot to continue a (longer) schedule from with a fresh optimizer")
    ap.add_argument("--rewarm-steps", type=int, default=100, help="lr ramp after --continue-from")
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")
    ap.add_argument("--lr", type=float, default=None, help="override train.lr (LR sweeps)")
    ap.add_argument("--name", default=None, help="override the run name (runs/<name>)")
    args = ap.parse_args()

    ddp = "WORLD_SIZE" in os.environ                   # launched by torchrun
    rank, world = (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])) if ddp else (0, 1)
    master = rank == 0
    if ddp:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
        if not master:
            builtins.print = lambda *a, **k: None      # only rank 0 prints

    cfg = json.load(open(args.config))
    t, d = cfg["train"], cfg["data"]
    if args.lr is not None:
        t["lr"] = args.lr
    if args.name:
        cfg["name"] = args.name
    torch.manual_seed(t.get("seed", 0))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_dir = os.path.join(RUNS_DIR, cfg["name"])
    os.makedirs(run_dir, exist_ok=True)

    tokens_per_micro = t["micro_batch"] * d["seq_len"]
    accum = max(1, t["batch_tokens"] // tokens_per_micro)
    if accum % world:
        raise ValueError(f"accum {accum} is not divisible by {world} GPUs")
    my_micros = list(range(rank, accum, world))        # this rank's micro-batches of every step
    continue_from = t.get("continue_from") and os.path.join(PROJECT_DIR, t["continue_from"])
    if continue_from:   # schedule of continue_tokens more tokens, counted from the snapshot's step
        base_step = torch.load(continue_from, map_location="cpu", weights_only=False)["step"]
        t["base_step"] = base_step
        t["steps"] = base_step + int(t["continue_tokens"] // (accum * tokens_per_micro))
    else:
        t["steps"] = int(t["total_tokens"] // (accum * tokens_per_micro))

    doc_mask = cfg["model"].get("doc_mask", False)
    eos_id = cfg["model"].get("eos_id", 0)
    if doc_mask and cfg["model"]["arch"] not in ("D", "E", "F"):
        raise ValueError("model.doc_mask is implemented for Arch D, E and F only")
    if ddp and t.get("mlm_prob", 0.0) > 0:
        raise ValueError("multi-GPU training does not support MLM batches (Arch C)")

    model, mcfg = build_model(cfg["model"])
    cfg["model"] = config_dict(mcfg)
    model = model.cuda()
    optimizer = make_optimizer(model, t)
    start_step = 0
    # Early stopping: stop after `early_stop_patience` evaluations in a row without the mean
    # validation loss improving by at least `early_stop_min_delta` (patience 0 = disabled).
    patience, min_delta = t.get("early_stop_patience", 0), t.get("early_stop_min_delta", 0.0)
    best_val, bad_evals = float("inf"), 0

    ckpt_latest = os.path.join(run_dir, "ckpt_latest.pt")
    stop_file = os.path.join(run_dir, "STOP")
    if master and os.path.exists(stop_file) and args.resume:
        os.remove(stop_file)                           # left over from the stop we are resuming after
    ckpt_best = os.path.join(run_dir, "ckpt_best.pt")
    if args.resume and os.path.exists(ckpt_latest):
        state = torch.load(ckpt_latest, map_location="cuda")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        best_val, bad_evals = state.get("best_val", float("inf")), state.get("bad_evals", 0)
        print(f"resumed from step {start_step} (best val {best_val:.4f}, {bad_evals} evals without improvement)")
    elif continue_from:
        state = torch.load(continue_from, map_location="cuda")
        model.load_state_dict(state["model"])
        start_step = state["step"]
        print(f"continuing from {continue_from} at step {start_step} for {t['steps'] - start_step} steps "
              f"(fresh optimizer, peak lr {t['lr']}, warmup {t['warmup_steps']}, doc_mask {doc_mask})")
    elif args.decay_from:
        state = torch.load(args.decay_from, map_location="cuda")
        model.load_state_dict(state["model"])
        start_step = state["step"]
        t["steps"] = start_step + int(t["decay_steps"])
        t["decay_frac"] = t["decay_steps"] / t["steps"]
        print(f"decay-only branch from step {start_step} for {t['decay_steps']} steps")
    elif args.continue_from:
        state = torch.load(args.continue_from, map_location="cuda")
        model.load_state_dict(state["model"])
        start_step = state["step"]
        t["continued_from"], t["rewarm_steps"] = args.continue_from, args.rewarm_steps
        print(f"continuing from {args.continue_from} at step {start_step} "
              f"(fresh optimizer, lr re-warmed over {args.rewarm_steps} steps)")

    if master:
        json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)

    sampler = MixedWindowSampler(d["tokenizer"], d["mix"], d["seq_len"], "train", seed=t.get("seed", 0))
    val_sets = {n: fixed_val_windows(d["tokenizer"], n, d["seq_len"], t.get("val_windows", 64)) for n in sampler.names}
    mlm_prob = t.get("mlm_prob", 0.0)
    mask_gen = torch.Generator().manual_seed(t.get("seed", 0))
    sd = t.get("single_doc_eval")
    if sd:
        sd_sets = {n: single_doc_val_windows(d["tokenizer"], n, d["seq_len"], sd.get("n_docs", 64)) for n in sampler.names}
        sd_ref = json.load(open(os.path.join(PROJECT_DIR, sd["reference"]))) if sd.get("reference") else {}
    fpt = flops_per_token(model, d["seq_len"])
    peak_flops = PEAK_BF16_FLOPS.get(torch.cuda.get_device_name(0))

    print(f"model {model.num_params() / 1e6:.1f}M params | arch {cfg['model']['arch']} | {world} GPU(s)")
    n_run = t["steps"] - t.get("base_step", 0)   # steps in this schedule (continued runs: after the snapshot)
    print(f"steps {n_run} x accum {accum} x micro {t['micro_batch']} x seq {d['seq_len']} "
          f"= {n_run * accum * tokens_per_micro / 1e9:.2f}B tokens")
    print("epochs per source:", {k: round(v, 2) for k, v in sampler.epochs_per_source(t.get("continue_tokens", t.get("total_tokens"))).items()})

    ddp_model = DDP(model, device_ids=[torch.cuda.current_device()]) if ddp else model
    train_model = torch.compile(ddp_model) if t.get("compile", True) else ddp_model
    log = open(os.path.join(run_dir, "metrics.jsonl"), "a") if master else None
    t0, tokens_seen = time.time(), 0

    val_bm_on = t.get("val_boundary_masked", False)

    def make_step(step):
        batches = [sampler.batch(t["micro_batch"], step, micro) for micro in my_micros]
        if doc_mask:
            batches = [(x, mask_boundary_labels(x, y, eos_id)) for x, y in batches]
        return batches
    stop = t["steps"] if not args.max_steps else min(t["steps"], start_step + args.max_steps)
    prefetch = Prefetcher(make_step, start_step, stop, t["prefetch"]) if t.get("prefetch") else None
    has_lambda = hasattr(model, "lambda_values") and len(model.lambda_values()) > 0
    torch.cuda.reset_peak_memory_stats()
    for step in range(start_step, t["steps"]):
        if continue_from:        # own schedule over the continued steps only
            lr = lr_at(step - t["base_step"], {**t, "steps": t["steps"] - t["base_step"]})
        else:
            lr = lr_at(step, t)
        if args.continue_from:   # fresh AdamW moments: ramp back up to the scheduled lr
            lr *= min(1.0, (step - start_step + 1) / args.rewarm_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        clm_loss = mlm_loss = 0.0
        n_clm = n_mlm = 0
        micro_batches = prefetch.get(step) if prefetch else make_step(step)
        if doc_mask:   # weight micro-batches by valid labels so the step loss is a per-token mean
            n_valid = [int((y != -100).sum()) for _, y in micro_batches]
            total_valid = sum(n_valid)
            if ddp:        # the mean is over the labels of all ranks
                tv = torch.tensor(total_valid, device="cuda")
                dist.all_reduce(tv)
                total_valid = int(tv)
        for micro, (x, y) in enumerate(micro_batches):
            use_mlm = mlm_prob > 0 and torch.rand(1, generator=mask_gen).item() < mlm_prob
            if use_mlm:   # Arch C: bidirectional pass predicting masked tokens
                x, y = mask_for_mlm(x, mask_id=2, vocab_size=cfg["model"]["vocab_size"],
                                    rate=t.get("mlm_mask_rate", 0.3), special_max_id=N_SPECIAL - 1,
                                    generator=mask_gen)
            x, y = x.pin_memory().cuda(non_blocking=True), y.pin_memory().cuda(non_blocking=True)
            # DDP: gradients are all-reduced only after the last local micro-batch
            sync = not ddp or micro == len(micro_batches) - 1
            with (contextlib.nullcontext() if sync else ddp_model.no_sync()):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if use_mlm:
                        _, loss = train_model(x, y, bidirectional=True)
                    else:
                        _, loss = train_model(x, y)
                weight = n_valid[micro] / total_valid if doc_mask else 1 / accum
                (loss * weight * world).backward()      # DDP averages gradients over the ranks
            if use_mlm:
                mlm_loss += loss.item(); n_mlm += 1
            else:
                clm_loss += loss.item() * (weight * accum if doc_mask else 1); n_clm += 1

        if ddp:            # step loss: add up the per-rank sums (same formula as on one GPU)
            lt = torch.tensor([clm_loss, n_clm], device="cuda", dtype=torch.float64)
            dist.all_reduce(lt)
            clm_loss, n_clm = lt[0].item(), int(lt[1])
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"]).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens_seen += accum * tokens_per_micro

        if master and step % t.get("log_every", 10) == 0:
            dt = time.time() - t0
            rec = {"step": step, "lr": lr, "loss": clm_loss / max(1, n_clm), "grad_norm": grad_norm,
                   "tok_per_s": tokens_seen / dt, "tokens": (step + 1) * accum * tokens_per_micro,
                   "mfu": tokens_seen / dt * fpt / (peak_flops * world) if peak_flops else None,
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
            if has_lambda:
                rec["lambda"] = model.lambda_values()
            if n_mlm:
                rec["mlm_loss"] = mlm_loss / n_mlm
            log.write(json.dumps(rec) + "\n"); log.flush()
            eta_h = (t["steps"] - step) * dt / max(1, step + 1 - start_step) / 3600
            extra = f" mfu {rec['mfu']:.1%}" if rec["mfu"] else ""
            extra += f" mem {rec['peak_mem_gb']:.1f}GB"
            if has_lambda:
                lams = [x for x in rec["lambda"] if x is not None]
                extra += f" lam {sum(lams) / max(1, len(lams)):.3f}"
            print(f"step {step}/{t['steps']} loss {rec['loss']:.4f} lr {lr:.2e} gn {grad_norm:.2f} "
                  f"{rec['tok_per_s'] / 1e3:.0f}k tok/s{extra} eta {eta_h:.1f}h", flush=True)

        last = step == t["steps"] - 1 or (args.max_steps and step + 1 - start_step >= args.max_steps)
        if master and sd and step + 1 in sd.get("steps", []):
            sdv = evaluate_single_doc(model, sd_sets, t["micro_batch"])
            ref = sd_ref.get(str(step + 1))
            tok_b = (step + 1) * accum * tokens_per_micro / 1e9
            log.write(json.dumps({"step": step, "single_doc_val": sdv, "reference": ref}) + "\n"); log.flush()
            print(f"  single-doc val after {step + 1} steps ({tok_b:.2f}B tokens): {sdv['mean']:.4f} | "
                  + " ".join(f"{k} {v:.3f}" for k, v in sdv.items() if k != "mean"), flush=True)
            if ref:
                print(f"  reference {sd_ref.get('name', '')} at the same step: {ref['mean']:.4f} | "
                      + " ".join(f"{k} {ref[k]:.3f}" for k in sdv if k != "mean")
                      + f" | difference {sdv['mean'] - ref['mean']:+.4f}", flush=True)
            if has_lambda:
                print("  lambda per layer: " + " ".join(f"{x:.2f}" for x in model.lambda_values() if x is not None),
                      flush=True)
        if master and (step % t["eval_every"] == 0 or last):
            val = evaluate(model, val_sets, t["micro_batch"], eos_id if doc_mask else None)
            rec = {"step": step, "val": val}
            if val_bm_on:   # boundary labels ignored in every run: comparable across doc_mask on/off
                rec["val_bm"] = val if doc_mask else evaluate(model, val_sets, t["micro_batch"], eos_id)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"  val loss {val['mean']:.4f} ppl {math.exp(val['mean']):.2f} | "
                  + " ".join(f"{k} {v:.3f}" for k, v in val.items() if k != "mean"), flush=True)
            if val_bm_on:
                print(f"  val_bm {rec['val_bm']['mean']:.4f}", flush=True)
            sel = rec["val_bm"] if val_bm_on else val
            if sel["mean"] < best_val - min_delta:
                best_val, bad_evals = sel["mean"], 0
                save(ckpt_best, model, optimizer, step + 1, cfg, full=False, extra={"val": val, **(
                    {"val_bm": rec["val_bm"]} if val_bm_on else {})})
                print(f"  new best val {best_val:.4f} -> ckpt_best.pt", flush=True)
            else:
                bad_evals += 1
                if patience and bad_evals >= patience:
                    print(f"  early stop: no improvement for {bad_evals} evals (best {best_val:.4f})", flush=True)
                    last = True
        if ddp and step % t["eval_every"] == 0:      # rank 0's early-stop decision reaches every rank
            flag = torch.tensor(int(bool(last)), device="cuda")
            dist.broadcast(flag, 0)
            last = bool(flag.item())
        stop_req = master and os.path.exists(stop_file)
        if ddp:                                        # every rank stops at the same step
            flag = torch.tensor(int(stop_req), device="cuda")
            dist.broadcast(flag, 0)
            stop_req = bool(flag.item())
        if stop_req and not last:
            if master:
                save(ckpt_latest, model, optimizer, step + 1, cfg,
                     extra={"best_val": best_val, "bad_evals": bad_evals})
                os.remove(stop_file)
            print(f"STOP file found: saved ckpt_latest.pt at step {step + 1}; continue with --resume", flush=True)
            break
        if master and ((step > start_step and step % t["save_every"] == 0) or last):
            save(ckpt_latest, model, optimizer, step + 1, cfg,
                 extra={"best_val": best_val, "bad_evals": bad_evals})
            save(os.path.join(run_dir, f"ckpt_step{step + 1}.pt"), model, optimizer, step + 1, cfg, full=False)
        if last:
            break
    print("done")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
