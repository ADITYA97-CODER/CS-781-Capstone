"""Does the model want to end the document inside its 128-word continuation?

  CUDA_VISIBLE_DEVICES=0 python phase1/diagnostics/check_eos_in_outputs.py \
      --ckpt runs/archD_sliding_72M_4B/ckpt_best.pt --ids 262 325 333 406 [--prompts local|official]

generate.py bans every special token and saves only decoded text, so the saved predictions can
never contain <|endoftext|>. This script regenerates the given prompts (one at a time, with the
sampling settings of --gen-args) and inspects the RAW token ids before decoding:
  free     special tokens allowed: is <|endoftext|> (or another special id) emitted, at which word?
  banned   specials suppressed as in generate.py: highest p(<|endoftext|>) along the continuation
           (teacher-forced over the generated text) and the word where it peaks.
"""

import argparse
import json
import os
import sys

import torch
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from generate import N_SPECIAL, PROJECT_DIR, PROMPT_FILES, load_model, word_start_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/archD_sliding_72M_4B/ckpt_best.pt")
ap.add_argument("--ids", nargs="+", default=["262", "325", "333", "406"])
ap.add_argument("--prompts", default="local")
ap.add_argument("--gen-args", default="outputs/phase1/archD_sliding_72M_local/generation_args.json",
                help="generation_args.json whose temperature/top-p/penalties are reused")
ap.add_argument("--max-new-tokens", type=int, default=320)
args = ap.parse_args()

g = json.load(open(os.path.join(PROJECT_DIR, args.gen_args)))
model, cfg = load_model(os.path.join(PROJECT_DIR, args.ckpt))
eos = model.cfg.eos_id if hasattr(model.cfg, "eos_id") else 0
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    os.path.join(PROJECT_DIR, "artifacts", "tokenizer", cfg["data"]["tokenizer"]))
starts_word = word_start_table(tokenizer).cuda()
prompts = json.load(open(PROMPT_FILES.get(args.prompts, args.prompts)))
max_prompt = model.cfg.max_seq_len - args.max_new_tokens
suppress = torch.arange(N_SPECIAL, device="cuda")
orig_head = model.lm_head.forward
print(f"ckpt {args.ckpt} | prompts {args.prompts} | T={g['temperature']} top-p={g['top_p']} "
      f"rep={g['repetition_penalty']} | <|endoftext|> id {eos}\n")


def word_index(gen_ids, pos):
    """1-based word number that token `pos` of the continuation belongs to."""
    return max(1, int(starts_word[torch.tensor(gen_ids[:pos + 1], device="cuda")].sum()))


def run(idx, banned):
    if banned:
        def head(h):
            logits = orig_head(h)
            logits[..., suppress] = float("-inf")
            return logits
        model.lm_head.forward = head
    else:
        model.lm_head.forward = orig_head
    torch.manual_seed(g["seed"])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        gen = model.generate(idx, args.max_new_tokens, temperature=g["temperature"], top_k=g["top_k"],
                             top_p=g["top_p"], repetition_penalty=g["repetition_penalty"],
                             no_repeat_ngram=g["no_repeat_ngram"],
                             stop_fn=lambda out: (out == eos).any(1) | (starts_word[out].sum(1) > 128))
    model.lm_head.forward = orig_head
    return gen[0].tolist()


for k in args.ids:
    ids = tokenizer(prompts[k]["text"], add_special_tokens=False)["input_ids"][-max_prompt:]
    idx = torch.tensor([ids], device="cuda")
    print(f"=== id {k} (prompt {len(ids)} tokens)")

    free = run(idx, banned=False)
    specials = [(p, t) for p, t in enumerate(free) if t < N_SPECIAL]
    print(f"  free:   raw ids contain <|endoftext|>: {eos in free}; special ids: "
          + (", ".join(f"{tokenizer.convert_ids_to_tokens(t)} at word {word_index(free, p)}" for p, t in specials)
             or "none"))
    if eos in free:
        cut = free.index(eos)
        print(f"          text before it: ...{tokenizer.decode(free[max(0, cut - 25):cut])!r}")

    banned = run(idx, banned=True)
    full = torch.tensor([ids + banned], device="cuda")[:, -model.cfg.max_seq_len:]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        p_eos = model(full)[0][0, -len(banned) - 1:-1].float().softmax(-1)[:, eos]   # before each generated token
    top = int(p_eos.argmax())
    print(f"  banned: max p(<|endoftext|>) {p_eos.max().item():.4f} before word {word_index(banned, top)} "
          f"(mean {p_eos.mean().item():.5f})")
    print(f"          text there: ...{tokenizer.decode(banned[max(0, top - 25):top])!r} | <here>\n")
