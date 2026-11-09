"""Generate 128-word continuations for Phase-1 prompts and package a submission.

Usage:
  python phase1/generate.py --ckpt runs/archD/ckpt_latest.pt --prompts official \
      --temperature 0.7 --top-p 0.9 --repetition-penalty 1.1 --out outputs/phase1/archD_t07
  python phase1/generate.py --ckpt ... --prompts local ...   # local held-out set, scored automatically
  python phase1/generate.py --ckpt ... --num-beams 4 --batch-size 32 ...   # beam search (rows = batch x beams)
  python phase1/generate.py --ckpt ... --prompt-boost --boost-b 1.0 --cad --cad-alpha 0.5 ...  # fact_decoding.py

Prompts are sorted by token length and batched; within a batch every prompt is
left-truncated to the shortest one, so no padding is needed (only a few tokens
from the *start* of a 600-900-token prompt are dropped). Special tokens are
suppressed, so the model cannot end early, and generation stops once every row
has more than 128 words. Output: <out>/prediction.json and <out>/submission.zip.

Optional (all off by default; with them off the behaviour above is unchanged):
  --prompt-boost / --cad   prompt-token boost and context-aware decoding (fact_decoding.py)
  --trunc-group N          left-truncate prompts per group of N sorted prompts (as batch size N would)
                           while running batches of --batch-size inside each group: a smaller batch
                           then sees exactly the same prompt tokens as the batch-128 runs
  --subset ids.json        generate only these ids (truncation still computed on the full prompt set)
  --max-mem-gb G           cap this process's CUDA memory (fails with OOM instead of growing)
  --prefix-facts F.json    {id: {"text": facts block}} (phase1/sft/facts.py format) placed before each prompt; the
                           block is never truncated: prompts are left-truncated to max_prompt - block length, and
                           batches (sorted by block + prompt length) are equalized by trimming PROMPT tokens only
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

from fact_decoding import (CLOSE_PHRASES, HEADER_PHRASES, HEADER_TOKENS, PromptBoost, guided_generate,
                           steer_citations, steer_phrases)
from models import build_model
from prompt_facts import load_or_build

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


def calibrate_js_ref(model, tokenizer, args, stop_fn, n_prompts=50, seed=0):
    """Median per-step JS(full, short) of fixed-alpha CAD on 50 fixed local prompts (cached per setting)."""
    import random
    key = (f"{os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))}_{os.path.basename(args.ckpt)}"
           f"_a{args.cad_alpha}_k{args.cad_short_len}_b{args.cad_beta}_t{args.temperature + args.cad_temp_offset}")
    path = os.path.join(PROJECT_DIR, "outputs", "phase1", "fact_decoding", "js_ref", key + ".json")
    if os.path.exists(path):
        return json.load(open(path))["js_ref"]
    local = json.load(open(PROMPT_FILES["local"]))
    ids = random.Random(seed).sample(sorted(local), n_prompts)
    enc = [tokenizer(local[k]["text"], add_special_tokens=False)["input_ids"] for k in ids]
    L = min(min(map(len, enc)), model.cfg.max_seq_len - args.max_new_tokens)
    idx = torch.tensor([e[-L:] for e in enc], device="cuda")
    record = []
    torch.manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        guided_generate(model, idx, args.max_new_tokens, temperature=args.temperature + args.cad_temp_offset,
                        top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty, stop_fn=stop_fn,
                        cad=dict(alpha=args.cad_alpha, short_len=args.cad_short_len, beta=args.cad_beta, record_js=record))
    js_ref = float(torch.tensor(record).median())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(dict(js_ref=js_ref, n_steps=len(record), prompts=ids, setting=key,
                   quantiles={q: float(torch.tensor(record).quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}),
              open(path, "w"), indent=1)
    return js_ref


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
    ap.add_argument("--prompt-boost", action="store_true", help="boost prompt specifics (fact_decoding.PromptBoost)")
    ap.add_argument("--boost-b", type=float, default=1.0, help="start boost on fact token ids")
    ap.add_argument("--boost-cont", type=float, default=4.0, help="continuation boost after a partial match")
    ap.add_argument("--facts", default=None, help="prompt_facts cache (built if missing); default under outputs/")
    ap.add_argument("--cad", action="store_true", help="context-aware decoding against a short-context run")
    ap.add_argument("--cad-alpha", type=float, default=0.5)
    ap.add_argument("--cad-short-len", type=int, default=32, help="prompt tokens the short run sees")
    ap.add_argument("--cad-beta", type=float, default=0.1, help="plausibility filter: p_full >= beta * max p_full")
    ap.add_argument("--steer-headers", action="store_true", help="penalize court headers / reporter lines")
    ap.add_argument("--steer-close", action="store_true", help="penalize closing formulas")
    ap.add_argument("--steer-cite", action="store_true", help="penalize citation reporters absent from the prompt")
    ap.add_argument("--steer-penalty", type=float, default=5.0, help="headers")
    ap.add_argument("--close-penalty", type=float, default=3.0)
    ap.add_argument("--cite-penalty", type=float, default=3.0)
    ap.add_argument("--no-repeat-gen-ngram", type=int, default=0,
                    help="ban repeating an n-gram of the generated text (prompt n-grams allowed)")
    ap.add_argument("--cad-adaptive", action="store_true", help="alpha_t = alpha * clamp(JS / js_ref, 0, 1)")
    ap.add_argument("--cad-js-ref", type=float, default=0.0,
                    help="JS reference for --cad-adaptive; 0 = median JS of a fixed-alpha calibration run on 50 local prompts")
    ap.add_argument("--cad-temp-offset", type=float, default=0.0, help="added to --temperature when --cad is on")
    ap.add_argument("--trunc-group", type=int, default=0, help="truncation group size (0 = batch size)")
    ap.add_argument("--subset", default=None, help="json list of prompt ids to generate")
    ap.add_argument("--max-mem-gb", type=float, default=0)
    ap.add_argument("--prefix-facts", default=None, help="json {id: {'text': facts block}} put before each prompt")
    args = ap.parse_args()

    if args.max_mem_gb:
        torch.cuda.set_per_process_memory_fraction(
            args.max_mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(args.seed)
    model, cfg = load_model(args.ckpt)
    tok_name = args.tokenizer or cfg["data"]["tokenizer"]
    tokenizer = PreTrainedTokenizerFast.from_pretrained(os.path.join(PROJECT_DIR, "artifacts", "tokenizer", tok_name))
    starts_word = word_start_table(tokenizer).cuda()

    prompts = json.load(open(PROMPT_FILES.get(args.prompts, args.prompts)))
    max_prompt = model.cfg.max_seq_len - args.max_new_tokens
    encoded = {k: tokenizer(v["text"], add_special_tokens=False)["input_ids"][-max_prompt:] for k, v in prompts.items()}
    prefix = {}
    if args.prefix_facts:
        assert not (args.trunc_group or args.subset or args.cad or args.prompt_boost or args.num_beams > 1), \
            "--prefix-facts is implemented for plain/steered sampling over the full prompt set"
        blocks = json.load(open(args.prefix_facts))
        prefix = {k: tokenizer(blocks[k]["text"], add_special_tokens=False)["input_ids"] for k in prompts}
        encoded = {k: v[len(v) - min(len(v), max_prompt - len(prefix[k])):] for k, v in encoded.items()}
    order = sorted(encoded, key=lambda k: len(encoded[k]) + len(prefix.get(k, ())))
    batches = [order[b:b + args.batch_size] for b in range(0, len(order), args.batch_size)]
    if args.trunc_group or args.subset:
        group = args.trunc_group or args.batch_size
        vis = {k: min(len(encoded[x]) for x in order[g:g + group]) for g in range(0, len(order), group)
               for k in order[g:g + group]}
        keep = set(json.load(open(args.subset))) if args.subset else set(order)
        batches, cur = [], []
        for k in (k for k in order if k in keep):         # batches never straddle a truncation group
            if cur and (len(cur) == args.batch_size or vis[k] != vis[cur[0]] or
                        order.index(k) // group != order.index(cur[0]) // group):
                batches.append(cur)
                cur = []
            cur.append(k)
        batches += [cur] if cur else []
        encoded = {k: v[-vis[k]:] for k, v in encoded.items()}
        prompts = {k: v for k, v in prompts.items() if k in keep}
    guided = (args.prompt_boost or args.cad or args.steer_headers or args.steer_close or args.steer_cite
              or args.no_repeat_gen_ngram > 0)
    assert not (guided and args.num_beams > 1), "--prompt-boost/--cad are implemented for sampling only"
    if args.prompt_boost:
        facts_path = args.facts or os.path.join(PROJECT_DIR, "outputs", "phase1", "fact_decoding",
                                                f"facts_{os.path.basename(PROMPT_FILES.get(args.prompts, args.prompts))}")
        facts = load_or_build(facts_path, prompts, tokenizer, tok_name)
    cad = dict(alpha=args.cad_alpha, short_len=args.cad_short_len, beta=args.cad_beta) if args.cad else None
    temperature = args.temperature + (args.cad_temp_offset if args.cad else 0.0)

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

    if args.cad and args.cad_adaptive:
        cad["js_ref"] = args.cad_js_ref or calibrate_js_ref(model, tokenizer, args, enough_words)
        torch.manual_seed(args.seed)                       # calibration must not shift the sampling stream
        print(f"adaptive CAD: js_ref = {cad['js_ref']:.5f}", flush=True)

    predictions, t0, done = {}, time.time(), 0
    visible = {}
    torch.cuda.reset_peak_memory_stats()
    for keys in batches:
        if prefix:                                         # equal total length; only prompt tokens are trimmed
            L = min(len(encoded[k]) + len(prefix[k]) for k in keys)
            rows = [prefix[k] + encoded[k][len(encoded[k]) - (L - len(prefix[k])):] for k in keys]
            visible.update({k: L - len(prefix[k]) for k in keys})
            idx = torch.tensor(rows, device="cuda")
        else:
            L = min(len(encoded[k]) for k in keys)
            idx = torch.tensor([encoded[k][-L:] for k in keys], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if guided:
                V = model.cfg.vocab_size
                boost = [PromptBoost([facts[k]["seqs"] for k in keys], [facts[k]["ids"] for k in keys],
                                     V, args.boost_b, args.boost_cont)] if args.prompt_boost else []
                if args.steer_headers:
                    boost.append(steer_phrases(tokenizer, len(keys), V, HEADER_PHRASES, HEADER_TOKENS, args.steer_penalty))
                if args.steer_close:
                    boost.append(steer_phrases(tokenizer, len(keys), V, CLOSE_PHRASES, (), args.close_penalty))
                if args.steer_cite:
                    boost.append(steer_citations(tokenizer, [prompts[k]["text"] for k in keys], V, args.cite_penalty))
                gen = guided_generate(model, idx, args.max_new_tokens, temperature=temperature,
                                      top_k=args.top_k, top_p=args.top_p,
                                      repetition_penalty=args.repetition_penalty,
                                      no_repeat_ngram=args.no_repeat_ngram, stop_fn=enough_words,
                                      boost=boost, cad=cad, no_repeat_gen_ngram=args.no_repeat_gen_ngram)
            elif args.num_beams > 1:
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
        done += len(keys)
        print(f"{done}/{len(prompts)} prompts ({time.time() - t0:.0f}s)", flush=True)
    stats = dict(n_prompts=len(prompts), seconds=time.time() - t0, sec_per_prompt=(time.time() - t0) / len(prompts),
                 peak_alloc_gb=torch.cuda.max_memory_allocated() / 2**30,
                 peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30)
    if prefix:
        stats.update(prefix_tokens_mean=sum(map(len, prefix.values())) / len(prefix),
                     visible_prompt_tokens_mean=sum(visible.values()) / len(visible))
        os.makedirs(args.out, exist_ok=True)
        json.dump(visible, open(os.path.join(args.out, "visible_prompt_tokens.json"), "w"))

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
    json.dump(stats, open(os.path.join(args.out, "generation_stats.json"), "w"), indent=1)
    print(f"wrote {pred_path} and submission.zip")

    if args.prompts == "local" and not args.subset:
        result = subprocess.run([sys.executable, EVALUATOR, "--truth", LOCAL_REF, "--prediction", pred_path],
                                capture_output=True, text=True)
        print(result.stdout, result.stderr)
        open(os.path.join(args.out, "local_scores.txt"), "w").write(result.stdout)


if __name__ == "__main__":
    main()
