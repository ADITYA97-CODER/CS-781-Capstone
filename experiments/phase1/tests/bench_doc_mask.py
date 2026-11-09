"""Test 8: time training steps of the real Arch D model with doc_mask on vs off.

  CUDA_VISIBLE_DEVICES=0 python phase1/tests/bench_doc_mask.py --doc-mask 0
  CUDA_VISIBLE_DEVICES=0 python phase1/tests/bench_doc_mask.py --doc-mask 1

Same batches, micro-batch and accumulation as configs/archD_4B_docmask*.json; the first
--warmup steps (torch.compile) are not timed.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import MixedWindowSampler, mask_boundary_labels  # noqa: E402
from models import build_model  # noqa: E402

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--doc-mask", type=int, required=True)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--config", default=os.path.join(PROJECT_DIR, "phase1", "configs", "archD_4B_docmask.json"))
args = ap.parse_args()

cfg = json.load(open(args.config))
t, d = cfg["train"], cfg["data"]
cfg["model"]["doc_mask"] = bool(args.doc_mask)
torch.backends.cuda.matmul.allow_tf32 = True
model = build_model(cfg["model"])[0].cuda()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
train_model = torch.compile(model)
sampler = MixedWindowSampler(d["tokenizer"], d["mix"], d["seq_len"], "train", seed=0)
accum = t["batch_tokens"] // (t["micro_batch"] * d["seq_len"])


def step(i):
    batches = [sampler.batch(t["micro_batch"], i, m) for m in range(accum)]
    if args.doc_mask:
        batches = [(x, mask_boundary_labels(x, y, 0)) for x, y in batches]
    n_valid = [int((y != -100).sum()) for _, y in batches]
    for (x, y), n in zip(batches, n_valid):
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = train_model(x, y)
        (loss * (n / sum(n_valid) if args.doc_mask else 1 / accum)).backward()
    opt.step(); opt.zero_grad(set_to_none=True)


t0 = time.time()
for i in range(args.warmup):
    step(i)
torch.cuda.synchronize()
compile_s = time.time() - t0
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
for i in range(args.warmup, args.warmup + args.steps):
    step(i)
torch.cuda.synchronize()
dt = time.time() - t0
tokens = args.steps * accum * t["micro_batch"] * d["seq_len"]
print(json.dumps({"doc_mask": bool(args.doc_mask), "steps": args.steps, "s_per_step": round(dt / args.steps, 3),
                  "tok_per_s": round(tokens / dt), "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 1),
                  "compile_and_warmup_s": round(compile_s, 1)}))
