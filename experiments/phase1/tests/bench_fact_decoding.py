"""Time and peak memory per prompt for baseline / boost / CAD / CAD+boost (test 7 of fact decoding).

  CUDA_VISIBLE_DEVICES=0 python phase1/tests/bench_fact_decoding.py --batch-sizes 16 32 64 --max-mem-gb 3.5

Uses one generate.py truncation group (128 length-sorted local prompts, left-truncated to the shortest)
and the submitted sampling settings (T 0.7, top-p 0.9, repetition penalty 1.1, 320 new tokens,
stop after 128 words). Peak memory = torch.cuda.max_memory_allocated during generation, model included.
A batch size that does not fit under --max-mem-gb is reported as OOM.
"""

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from fact_decoding import PromptBoost, guided_generate  # noqa: E402
from generate import PROJECT_DIR, TARGET_WORDS, load_model, word_start_table  # noqa: E402
from test_fact_decoding import FACTS, PROMPTS, patch_head  # noqa: E402

MODES = {"baseline": dict(), "boost": dict(boost=True), "cad": dict(cad=True), "cad+boost": dict(boost=True, cad=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(PROJECT_DIR, "runs", "archD_4B_docmask", "ckpt_best.pt"))
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--group", type=int, default=3, help="which 128-prompt truncation group (length-sorted)")
    ap.add_argument("--max-mem-gb", type=float, default=3.5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.max_mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)

    from transformers import PreTrainedTokenizerFast
    model, cfg = load_model(a.ckpt)
    patch_head(model)
    tok = PreTrainedTokenizerFast.from_pretrained(os.path.join(PROJECT_DIR, "artifacts", "tokenizer",
                                                               cfg["data"]["tokenizer"]))
    starts = word_start_table(tok).cuda()
    stop = lambda g: starts[g].sum(1) > TARGET_WORDS
    prompts, facts = json.load(open(PROMPTS)), json.load(open(FACTS))
    enc = {k: tok(v["text"], add_special_tokens=False)["input_ids"][-(1536 - 320):] for k, v in prompts.items()}
    order = sorted(enc, key=lambda k: len(enc[k]))
    grp = order[a.group * 128:(a.group + 1) * 128]
    L = min(len(enc[k]) for k in grp)
    ids = grp[:a.n_prompts]
    print(f"{len(ids)} prompts, prompt length {L} tokens (group {a.group})")
    V = model.cfg.vocab_size
    rows = []
    for bs in a.batch_sizes:
        for mode, m in MODES.items():
            torch.manual_seed(1)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0, steps = time.time(), 0
            try:
                for b in range(0, len(ids), bs):
                    keys = ids[b:b + bs]
                    idx = torch.tensor([enc[k][-L:] for k in keys], device="cuda")
                    boost = PromptBoost([facts[k]["seqs"] for k in keys], [facts[k]["ids"] for k in keys], V) \
                        if m.get("boost") else None
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        g = guided_generate(model, idx, 320, temperature=0.7, top_p=0.9, repetition_penalty=1.1,
                                            stop_fn=stop, boost=boost,
                                            cad=dict(alpha=0.5, short_len=32, beta=0.1) if m.get("cad") else None)
                    steps += g.size(1)
                torch.cuda.synchronize()
                dt = time.time() - t0
                r = dict(batch=bs, mode=mode, sec_per_prompt=dt / len(ids), ms_per_step=1000 * dt / steps,
                         peak_gb=torch.cuda.max_memory_allocated() / 2**30)
            except torch.OutOfMemoryError:
                r = dict(batch=bs, mode=mode, sec_per_prompt=None, ms_per_step=None, peak_gb=f">{a.max_mem_gb}")
            rows.append(r)
            print(json.dumps(r), flush=True)
    print(f"\n| batch | mode | s / prompt | ms / step | peak GB |\n|---|---|---|---|---|")
    for r in rows:
        f = lambda v, p: "OOM" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else v)
        print(f"| {r['batch']} | {r['mode']} | {f(r['sec_per_prompt'], 3)} | {f(r['ms_per_step'], 1)} | {f(r['peak_gb'], 2)} |")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
