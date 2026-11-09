"""Generate 128-word continuations for Phase-1 prompts and package a submission.

Usage:
  python phase1/generate.py --ckpt runs/archD/ckpt_latest.pt --prompts official \
      --temperature 0.7 --top-p 0.9 --repetition-penalty 1.1 --out outputs/phase1/archD_t07
  python phase1/generate.py --ckpt ... --prompts local ...   # local held-out set, scored automatically
  python phase1/generate.py --ckpt ... --num-beams 4 --batch-size 32 ...   # beam search (rows = batch x beams)

Prompts are sorted by token length and batched; within a batch every prompt is
left-truncated to the shortest one, so no padding is needed (only a few tokens
from the *start* of a 600-900-token prompt are dropped). Special tokens are
suppressed, so the model cannot end early, and generation stops once every row
has more than 128 words. Output: <out>/prediction.json and <out>/submission.zip.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile

import torch
from transformers import PreTrainedTokenizerFast

from models import build_model

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROMPT_FILES = {
    "official": os.path.join(PROJECT_DIR, "data", "Phase_1_Pretraining", "Test", "test.json"),
    "local": os.path.join(PROJECT_DIR, "data", "corpus", "eval", "local_test.json"),
}
LOCAL_REF = os.path.join(PROJECT_DIR, "data", "corpus", "eval", "local_ref.json")
EVALUATOR = os.path.join(PROJECT_DIR, "official_eval", "Evaluation Script", "evaluate_phase1.py")
TARGET_WORDS = 128
N_SPECIAL = 105


def load_model(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    model, cfg = build_model(state["config"]["model"])
    model.load_state_dict(state["model"])
    return model.cuda().eval(), state["config"]


def word_start_table(tokenizer):
    """Bool per token id: True if the token begins a new whitespace-separated word."""
    table = torch.zeros(len(tokenizer), dtype=torch.bool)
    for tok, i in tokenizer.get_vocab().items():
        table[i] = tok.startswith("Ġ")    # byte-level BPE marks a leading space with 'Ġ'
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompts", default="local", help="'official', 'local', or a path to a prompts json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--no-repeat-ngram", type=int, default=0)
    ap.add_argument("--num-beams", type=int, default=1, help=">1 = beam search (temperature/top-k/top-p ignored)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, cfg = load_model(args.ckpt)
    tok_name = args.tokenizer or cfg["data"]["tokenizer"]
    tokenizer = PreTrainedTokenizerFast.from_pretrained(os.path.join(PROJECT_DIR, "artifacts", "tokenizer", tok_name))
    starts_word = word_start_table(tokenizer).cuda()

    prompts = json.load(open(PROMPT_FILES.get(args.prompts, args.prompts)))
    max_prompt = model.cfg.max_seq_len - args.max_new_tokens
    encoded = {k: tokenizer(v["text"], add_special_tokens=False)["input_ids"][-max_prompt:] for k, v in prompts.items()}
    order = sorted(encoded, key=lambda k: len(encoded[k]))

    # Never emit special tokens (end-of-text, pad, mask, sentinels): continuations must reach 128 words.
    suppress = torch.arange(N_SPECIAL, device="cuda")
    orig_head = model.lm_head.forward

    def head_without_specials(h):
        logits = orig_head(h)
        logits[..., suppress] = float("-inf")
        return logits
    model.lm_head.forward = head_without_specials

    def enough_words(gen):  # the 129th word has started => the first 128 words are complete
        return starts_word[gen].sum(1) > TARGET_WORDS

    predictions, t0 = {}, time.time()
    for b in range(0, len(order), args.batch_size):
        keys = order[b:b + args.batch_size]
        L = min(len(encoded[k]) for k in keys)
        idx = torch.tensor([encoded[k][-L:] for k in keys], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if args.num_beams > 1:
                gen = model.beam_generate(idx, args.max_new_tokens, num_beams=args.num_beams,
                                          repetition_penalty=args.repetition_penalty,
                                          no_repeat_ngram=args.no_repeat_ngram, stop_fn=enough_words)
            else:
                gen = model.generate(idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
                                     top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                                     no_repeat_ngram=args.no_repeat_ngram, stop_fn=enough_words)
        for k, row in zip(keys, gen.tolist()):
            words = tokenizer.decode(row, skip_special_tokens=True).split()
            predictions[k] = {"text": " ".join(words[:TARGET_WORDS])}
        print(f"{min(b + args.batch_size, len(order))}/{len(order)} prompts ({time.time() - t0:.0f}s)", flush=True)

    short = [k for k, v in predictions.items() if len(v["text"].split()) < TARGET_WORDS]
    if short:
        print(f"WARNING: {len(short)} continuations have fewer than {TARGET_WORDS} words "
              f"(raise --max-new-tokens): {short[:10]}")

    os.makedirs(args.out, exist_ok=True)
    pred_path = os.path.join(args.out, "prediction.json")
    ordered = {k: predictions[k] for k in prompts}          # keep the official id order
    json.dump(ordered, open(pred_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    with zipfile.ZipFile(os.path.join(args.out, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, arcname="prediction.json")
    json.dump(vars(args), open(os.path.join(args.out, "generation_args.json"), "w"), indent=1)
    print(f"wrote {pred_path} and submission.zip")

    if args.prompts == "local":
        result = subprocess.run([sys.executable, EVALUATOR, "--truth", LOCAL_REF, "--prediction", pred_path],
                                capture_output=True, text=True)
        print(result.stdout, result.stderr)
        open(os.path.join(args.out, "local_scores.txt"), "w").write(result.stdout)


if __name__ == "__main__":
    main()
