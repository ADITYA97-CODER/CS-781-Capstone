"""Sample 128-word continuations from the HF LLaMA model (same prompt handling as the notebook).

Usage:
  python phase1/mbr/llama_sample.py --prompts local --seed 1 --out outputs/phase1/mbr/llama_s1.json
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(PROJECT_DIR, "outputs", "llama_phase1", "legal-llama-final")
PROMPT_FILES = {"official": os.path.join(PROJECT_DIR, "data1", "Phase_1_Pretraining", "Test", "test.json"),
                "local": os.path.join(PROJECT_DIR, "data", "corpus", "eval", "local_test.json")}
CONTEXT, MAX_NEW, TARGET_WORDS = 1536, 384, 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="local")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.padding_side, tok.truncation_side = "left", "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16).cuda().eval()
    prompts = json.load(open(PROMPT_FILES.get(args.prompts, args.prompts)))
    items = list(prompts.items())
    torch.manual_seed(args.seed)

    preds = {}
    for s in range(0, len(items), args.batch_size):
        batch = items[s:s + args.batch_size]
        enc = tok([r["text"] for _, r in batch], return_tensors="pt", padding=True, truncation=True,
                  max_length=CONTEXT - MAX_NEW).to("cuda")
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=args.temperature,
                                 top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                                 pad_token_id=tok.pad_token_id, suppress_tokens=list(range(105)))
        for (k, _), row in zip(batch, out[:, enc["input_ids"].shape[1]:]):
            words = tok.decode(row, skip_special_tokens=True).split()[:TARGET_WORDS]
            preds[k] = {"text": " ".join(words) or "."}
        print(f"{s + len(batch)}/{len(items)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(preds, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
