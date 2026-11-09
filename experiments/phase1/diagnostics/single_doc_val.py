"""Validation loss on single-document rows (each row = one validation document from its first token,
see data.single_doc_val_windows) for snapshots of a run, saved as the reference that train.py's
`single_doc_eval` prints next to a new run's loss at the same step (same batch size -> same tokens).

  CUDA_VISIBLE_DEVICES=0 python phase1/diagnostics/single_doc_val.py --run runs/archD_sliding_72M \
      --steps 1001 2001 --out runs/archD_sliding_72M/single_doc_val.json

Arch D's 4B run (archD_sliding_72M_4B) continued the 2B run from step 3001, so steps 1001 and 2001
(0.49B and 0.98B tokens) of the 2B run are also its snapshots at those token counts.
"""

import argparse
import json
import os
import sys

import torch

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_DIR, "phase1"))
from data import single_doc_val_windows  # noqa: E402
from models import build_model  # noqa: E402
from train import evaluate_single_doc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", nargs="+", type=int, required=True)
    ap.add_argument("--n-docs", type=int, default=64)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(PROJECT_DIR, args.run, "config.json")))
    d = cfg["data"]
    sets = {n: single_doc_val_windows(d["tokenizer"], n, d["seq_len"], args.n_docs) for n in d["mix"] if d["mix"][n] > 0}
    out = {"name": cfg["name"], "n_docs": args.n_docs,
           "batch_tokens": cfg["train"]["batch_tokens"]}
    for step in args.steps:
        state = torch.load(os.path.join(PROJECT_DIR, args.run, f"ckpt_step{step}.pt"), map_location="cpu", weights_only=False)
        model = build_model(state["config"]["model"])[0]
        model.load_state_dict(state["model"])
        res = evaluate_single_doc(model.cuda(), sets, args.micro_batch)
        res["tokens"] = step * cfg["train"]["batch_tokens"]
        out[str(step)] = res
        print(step, json.dumps(res), flush=True)
    json.dump(out, open(os.path.join(PROJECT_DIR, args.out), "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
