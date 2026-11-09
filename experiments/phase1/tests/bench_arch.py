"""Test 11: training throughput and memory of any config (real data, compiled, bf16, fused AdamW).

  CUDA_VISIBLE_DEVICES=0 python phase1/tests/bench_arch.py --config phase1/configs/archE.json --micro-batch 16
  CUDA_VISIBLE_DEVICES=0 python phase1/tests/bench_arch.py --config phase1/configs/archE.json --probe 8,12,16,20,24,32
      # memory probe: each micro-batch size in a fresh process; prints the largest that fits

Each timed step is `--accum` micro-batches + one optimizer step, built exactly as train.py builds
them (doc_mask labels, background prefetch); the first --warmup steps (compilation) are not timed.
GPU utilisation is sampled with nvidia-smi during the timed steps.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_DIR, "phase1"))


def probe(args):
    fits = None
    for mb in [int(x) for x in args.probe.split(",")]:
        cmd = [sys.executable, __file__, "--config", args.config, "--micro-batch", str(mb), "--steps", "2",
               "--warmup", "1", "--accum", "1"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        if r.returncode == 0 and line:
            res = json.loads(line[-1])
            print(f"micro_batch {mb}: fits, peak {res['peak_mem_GB']} GB", flush=True)
            fits = mb
        else:
            oom = "out of memory" in (r.stderr + r.stdout).lower()
            print(f"micro_batch {mb}: {'OOM' if oom else 'FAILED'}", flush=True)
            if not oom:
                print(r.stderr[-2000:])
            break
    print(json.dumps({"largest_micro_batch_that_fits": fits}))


def gpu_util_sampler(stop, samples):
    idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    while not stop.is_set():
        try:
            out = subprocess.run(["nvidia-smi", "-i", idx, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            samples.append(float(out))
        except Exception:  # noqa: BLE001
            pass
        stop.wait(1.0)


def bench(args):
    import torch
    from data import MixedWindowSampler, Prefetcher, mask_boundary_labels
    from models import build_model
    from train import PEAK_BF16_FLOPS, flops_per_token, make_optimizer

    cfg = json.load(open(args.config))
    t, d = cfg["train"], cfg["data"]
    if args.doc_mask is not None:
        cfg["model"]["doc_mask"] = bool(args.doc_mask)
    doc_mask, eos = cfg["model"].get("doc_mask", False), cfg["model"].get("eos_id", 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    model = build_model(cfg["model"])[0].cuda()
    opt = make_optimizer(model, t)
    train_model = torch.compile(model)
    sampler = MixedWindowSampler(d["tokenizer"], d["mix"], d["seq_len"], "train", seed=0)
    mb, accum = args.micro_batch, args.accum

    def make_step(i):
        b = [sampler.batch(mb, i, m) for m in range(accum)]
        return [(x, mask_boundary_labels(x, y, eos)) for x, y in b] if doc_mask else b
    n_steps = args.warmup + args.steps
    pf = Prefetcher(make_step, 0, n_steps, 2)

    def step(i):
        batches = pf.get(i)
        n_valid = [int((y != -100).sum()) for _, y in batches]
        for (x, y), n in zip(batches, n_valid):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = train_model(x, y)
            (loss * (n / sum(n_valid))).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    t0 = time.time()
    for i in range(args.warmup):
        step(i)
    torch.cuda.synchronize()
    compile_s = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    stop, util = threading.Event(), []
    sampler_thread = threading.Thread(target=gpu_util_sampler, args=(stop, util), daemon=True)
    sampler_thread.start()
    t0 = time.time()
    for i in range(args.warmup, n_steps):
        loss = step(i)
    torch.cuda.synchronize()
    dt = time.time() - t0
    stop.set()
    tokens = args.steps * accum * mb * d["seq_len"]
    tps = tokens / dt
    fpt = flops_per_token(model, d["seq_len"])
    peak = PEAK_BF16_FLOPS.get(torch.cuda.get_device_name(0))
    print(json.dumps({"config": os.path.basename(args.config), "arch": cfg["model"]["arch"], "doc_mask": doc_mask,
                      "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                      "micro_batch": mb, "accum": accum, "steps": args.steps, "tok_per_s": round(tps),
                      "GFLOP_per_token": round(fpt / 1e9, 3), "mfu": round(tps * fpt / peak, 3) if peak else None,
                      "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 1),
                      "gpu_util_mean": round(sum(util) / len(util), 1) if util else None,
                      "compile_and_warmup_s": round(compile_s, 1), "last_loss": round(loss.item(), 3)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--doc-mask", type=int, default=None, help="override model.doc_mask")
    ap.add_argument("--probe", default=None, help="comma-separated micro-batch sizes to try")
    a = ap.parse_args()
    probe(a) if a.probe else bench(a)
