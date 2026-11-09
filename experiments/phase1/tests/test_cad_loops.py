"""Tests for the CAD loop fixes (mbr_select --loop-filter, --no-repeat-gen-ngram, --cad-adaptive, --cad-temp-offset).

  CUDA_VISIBLE_DEVICES=1 python phase1/tests/test_cad_loops.py [--max-mem-gb 4]

1  flags off: generate.py == pre-change generate.py (20 prompts, byte-identical); mbr_select without
   --loop-filter reproduces the stored Stage-B CAD MBR file byte-for-byte (990 prompts)
2  loop filter on synthetic candidates
3  no_repeat_gen_ngram: blocks a 5-gram repeated inside the output, allows a 5-gram that occurs in the prompt;
   end-to-end on the model: no repeated generated 5-gram
4  cad_adaptive: js_ref = inf == normal decoding (beta 0); JS always >= js_ref (js_ref -> 0) == fixed-alpha CAD
The rest of the CAD/boost suite is tests/test_fact_decoding.py.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "mbr"))
sys.path.insert(0, HERE)
from fact_decoding import ban_generated_repeats, block_repeated_ngrams, guided_generate, js_divergence  # noqa: E402
from generate import PROJECT_DIR, TARGET_WORDS, load_model, word_start_table  # noqa: E402
from mbr_select import choose, repetition  # noqa: E402
from test_fact_decoding import GEN, PROMPTS, batch_of, check, patch_head, results, run, run_ref  # noqa: E402

B_DIR = os.path.join(PROJECT_DIR, "outputs", "phase1", "fact_decoding", "B")


def test_flags_off_generate(ckpt, ids, prompts, max_mem):
    with tempfile.TemporaryDirectory() as d:
        pf = os.path.join(d, "prompts.json")
        json.dump({k: prompts[k] for k in ids}, open(pf, "w"))
        src = open(os.path.join(HERE, "reference_generate_pre_cad.py")).read()
        src = src.replace('os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))', repr(PROJECT_DIR))
        open(os.path.join(d, "ref.py"), "w").write(src)
        outs = {}
        for name, script in [("pre-change", os.path.join(d, "ref.py")), ("current", os.path.join(HERE, "..", "generate.py"))]:
            o = os.path.join(d, name)
            subprocess.run([sys.executable, script, "--ckpt", ckpt, "--prompts", pf, "--out", o, "--seed", "3",
                            "--repetition-penalty", "1.1", "--batch-size", "128"], check=True, capture_output=True,
                           env=dict(os.environ, PYTHONPATH=os.path.join(HERE, "..")))
            outs[name] = open(os.path.join(o, "prediction.json"), "rb").read()
        check("1a generate.py, all new flags off == pre-change generate.py (20 prompts, byte-identical)",
              outs["pre-change"] == outs["current"])


def test_flags_off_mbr():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "mbr.json")
        cands = [os.path.join(B_DIR, "cad_a0.5", f"s{s}", "prediction.json") for s in range(1, 5)]
        subprocess.run([sys.executable, os.path.join(HERE, "..", "mbr", "mbr_select.py"), "--cands", *cands,
                        "--out", out, "--batch-size", "64"], check=True, capture_output=True)
        same = open(out, "rb").read() == open(os.path.join(B_DIR, "cad_a0.5", "mbr.json"), "rb").read()
        check("1b mbr_select without --loop-filter == stored Stage-B CAD MBR file (990 prompts, byte-identical)", same)


def test_loop_filter():
    loop = " ".join(["the appellant filed the appeal"] * 26)[:]          # 130 words, ~80% repeated 4-grams
    ok = lambda i: " ".join(f"w{i}_{j}" for j in range(128))
    assert repetition(loop) > 0.25 and repetition(ok(1)) == 0
    texts = [loop, ok(1), ok(2), ok(3)]
    f = {(0, 1): .95, (0, 2): .94, (0, 3): .93, (1, 2): .80, (1, 3): .85, (2, 3): .90}
    b0, _ = choose(texts, f)
    b1, cons = choose(texts, f, loop_filter=True)
    check("2a without the filter the looping candidate wins on consensus (it agrees with everyone)", b0 == 0)
    check("2b with the filter it is dropped and consensus is recomputed among the rest (cand 3: .85+.90)",
          b1 == 3 and abs(cons - (0.85 + 0.90) / 2) < 1e-9)
    loops = [" ".join(["a b c d"] * 32), " ".join(["a b c d e f g h"] * 16), " ".join(["x y"] * 64)]
    reps = [repetition(t) for t in loops]
    b2, _ = choose(loops, {(0, 1): .9, (0, 2): .9, (1, 2): .9}, loop_filter=True)
    check("2c all candidates loop -> the one with the lowest repetition score is kept",
          all(r > 0.25 for r in reps) and b2 == min(range(3), key=reps.__getitem__), f"reps {[round(r, 3) for r in reps]}")
    short = " ".join(f"s{j}" for j in range(100))
    b3, _ = choose([loop, short, ok(2)], {(0, 1): .9, (0, 2): .5, (1, 2): .99}, loop_filter=True)
    check("2d the 128-word preference still applies inside the filtered pool", b3 == 2)


def test_no_repeat_gen():
    V = 12
    prompt = torch.tensor([[1, 2, 3, 4, 5, 9]])
    gen = torch.tensor([[7, 1, 2, 3, 4]])                       # about to copy the prompt 5-gram 1 2 3 4 5
    out = ban_generated_repeats(torch.zeros(1, V), gen, 5)
    old = block_repeated_ngrams(torch.zeros(1, V), torch.cat([prompt, gen], 1), 5)
    check("3a a 5-gram that occurs in the prompt (not yet in the output) is allowed",
          torch.isfinite(out[0, 5]).item() and torch.isinf(old[0, 5]).item(),
          "the existing --no-repeat-ngram, which also looks at the prompt, would ban it")
    gen = torch.tensor([[1, 2, 3, 4, 5, 8, 1, 2, 3, 4]])          # 1 2 3 4 5 already generated once
    out = ban_generated_repeats(torch.zeros(1, V), gen, 5)
    check("3b a 5-gram already generated cannot be generated again (only that token is banned)",
          torch.isinf(out[0, 5]).item() and torch.isfinite(out[0, [i for i in range(V) if i != 5]]).all().item())
    check("3c fewer than n generated tokens: nothing banned",
          torch.isfinite(ban_generated_repeats(torch.zeros(1, V), torch.tensor([[1, 2, 3]]), 5)).all().item())


def test_model(model, idx, stop, prompt_ids_list):
    st = {}
    gen = run(model, idx, 11, max_new_tokens=320, stop_fn=stop, cad=dict(alpha=1.0, short_len=32, beta=0.1),
              no_repeat_gen_ngram=5, stats=st, **GEN)
    print(f"   info: steps where a row had every token banned and fell back to plain logits: {int(st.get('fallback_steps', 0))}")
    rep5 = copied = 0
    for r, row in enumerate(gen.tolist()):
        L = next((j for j in range(len(row)) if stop(gen[r:r + 1, :j + 1]).item()), len(row) - 1) + 1
        g = [tuple(row[i:i + 5]) for i in range(L - 4)]
        rep5 += len(g) - len(set(g))
        p5 = {tuple(prompt_ids_list[r][i:i + 5]) for i in range(len(prompt_ids_list[r]) - 4)}
        copied += any(x in p5 for x in g)
    check("3d end-to-end (CAD alpha 1.0 + no_repeat_gen_ngram 5): no repeated generated 5-gram", rep5 == 0,
          f"{copied}/{len(gen)} outputs still copy at least one prompt 5-gram")

    x = torch.randn(4, 50, device="cuda")
    js_same, js_far = js_divergence(x, x), js_divergence(x * 50, -x * 50)
    check("4a JS(p, p) = 0 and JS <= ln 2", js_same.abs().max().item() < 1e-6 and js_far.max().item() <= 0.6932)
    a = run_ref(model, idx, 5, max_new_tokens=320, stop_fn=stop, **GEN)
    b = run(model, idx, 5, max_new_tokens=320, stop_fn=stop,
            cad=dict(alpha=0.5, short_len=32, beta=0.0, js_ref=float("inf")), **GEN)
    check("4b cad_adaptive with js_ref = inf (beta 0) == normal decoding", torch.equal(a, b))
    c = run(model, idx, 5, max_new_tokens=320, stop_fn=stop, cad=dict(alpha=0.5, short_len=32, beta=0.1), **GEN)
    d = run(model, idx, 5, max_new_tokens=320, stop_fn=stop,
            cad=dict(alpha=0.5, short_len=32, beta=0.1, js_ref=1e-30), **GEN)
    check("4c cad_adaptive with JS always >= js_ref (js_ref = 1e-30) == fixed-alpha CAD", torch.equal(c, d))
    rec = []
    run(model, idx, 5, max_new_tokens=320, stop_fn=stop, cad=dict(alpha=0.5, short_len=32, beta=0.1, record_js=rec), **GEN)
    e = run(model, idx, 5, max_new_tokens=320, stop_fn=stop, cad=dict(alpha=0.5, short_len=32, beta=0.1), **GEN)
    check("4d recording JS does not change the output", torch.equal(c, e), f"{len(rec)} JS values, median {torch.tensor(rec).median():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(PROJECT_DIR, "runs", "archD_4B_docmask", "ckpt_best.pt"))
    ap.add_argument("--max-mem-gb", type=float, default=4.0)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.max_mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    from transformers import PreTrainedTokenizerFast
    prompts = json.load(open(PROMPTS))
    ids = random.Random(0).sample(sorted(prompts), 20)
    test_loop_filter()
    test_no_repeat_gen()
    test_flags_off_generate(a.ckpt, ids, prompts, a.max_mem_gb)
    test_flags_off_mbr()
    model, cfg = load_model(a.ckpt)
    patch_head(model)
    tok = PreTrainedTokenizerFast.from_pretrained(os.path.join(PROJECT_DIR, "artifacts", "tokenizer", cfg["data"]["tokenizer"]))
    starts = word_start_table(tok).cuda()
    stop = lambda g: starts[g].sum(1) > TARGET_WORDS
    idx = batch_of(tok, prompts, ids, model.cfg.max_seq_len - 320)
    test_model(model, idx, stop, idx.tolist())
    print(f"\npeak CUDA memory allocated by the tests: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    print(f"{sum(ok for _, ok in results)}/{len(results)} checks passed")
    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == "__main__":
    main()
