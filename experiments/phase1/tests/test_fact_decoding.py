"""Tests for prompt-token boost and context-aware decoding (fact_decoding.py, generate.py flags).

  CUDA_VISIBLE_DEVICES=0 python phase1/tests/test_fact_decoding.py [--ckpt runs/archD_4B_docmask/ckpt_best.pt]

1  flags off: generate.py output == pre-change generate.py (tests/reference_generate_pre_cad.py),
   and guided_generate without boost/CAD == model.generate (same seed)
2  CAD alpha = 0 == normal decoding
3  CAD with short_len >= prompt length: logits_short == logits_full, output == normal decoding
4  incremental short-cache logits == one full forward over (last k prompt tokens + generated tokens)
5  PromptBoost on toy logits (start boost, longest continuation match, per-row facts)
6  batched generation with a different fact list per prompt == each prompt alone
Timing / peak memory (item 7) is tests/bench_fact_decoding.py.
The process caps its CUDA memory at --max-mem-gb (default 3) so it cannot squeeze a job sharing the GPU.
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
from fact_decoding import PromptBoost, cad_combine, guided_generate  # noqa: E402
from generate import N_SPECIAL, PROJECT_DIR, TARGET_WORDS, load_model, word_start_table  # noqa: E402

PROMPTS = os.path.join(PROJECT_DIR, "data", "corpus", "eval", "local_test.json")
FACTS = os.path.join(PROJECT_DIR, "outputs", "phase1", "fact_decoding", "facts_local_test.json")
GEN = dict(temperature=0.7, top_k=0, top_p=0.9, repetition_penalty=1.1)
results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)


def patch_head(model):
    """generate.py's special-token ban, so the tests decode exactly like generate.py."""
    orig = model.lm_head.forward
    suppress = torch.arange(N_SPECIAL, device="cuda")

    def head(h):
        logits = orig(h)
        logits[..., suppress] = float("-inf")
        return logits
    model.lm_head.forward = head


def batch_of(tok, prompts, ids, max_prompt):
    enc = [tok(prompts[k]["text"], add_special_tokens=False)["input_ids"][-max_prompt:] for k in ids]
    L = min(map(len, enc))
    return torch.tensor([e[-L:] for e in enc], device="cuda")


def run(model, idx, seed, bf16=True, **kw):
    torch.manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        return guided_generate(model, idx, **kw)


def run_ref(model, idx, seed, **kw):
    torch.manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model.generate(idx, **kw)


# ----------------------------------------------------------------------------- 1


def test_generate_py_unchanged(ckpt, ids, prompts):
    with tempfile.TemporaryDirectory() as d:
        pf = os.path.join(d, "prompts.json")
        json.dump({k: prompts[k] for k in ids}, open(pf, "w"))
        outs = {}
        ref_src = open(os.path.join(HERE, "reference_generate_pre_cad.py")).read()   # pin its PROJECT_DIR
        ref_src = ref_src.replace('os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))', repr(PROJECT_DIR))
        open(os.path.join(d, "reference_generate.py"), "w").write(ref_src)
        for name, script in [("pre-change", os.path.join(d, "reference_generate.py")),
                             ("current", os.path.join(HERE, "..", "generate.py"))]:
            o = os.path.join(d, name)
            env = dict(os.environ, PYTHONPATH=os.path.join(HERE, ".."))
            subprocess.run([sys.executable, script, "--ckpt", ckpt, "--prompts", pf, "--out", o, "--seed", "3",
                            "--repetition-penalty", "1.1", "--batch-size", "128"],
                           check=True, env=env, capture_output=True)
            outs[name] = open(os.path.join(o, "prediction.json"), "rb").read()
        check("1a generate.py, flags off == pre-change generate.py (20 prompts, seed 3, byte-identical json)",
              outs["pre-change"] == outs["current"], f"{len(outs['current'])} bytes")


def test_loop_equivalence(model, idx, stop):
    a = run_ref(model, idx, 5, max_new_tokens=320, stop_fn=stop, **GEN)
    b = run(model, idx, 5, max_new_tokens=320, stop_fn=stop, **GEN)
    check("1b guided_generate without boost/CAD == model.generate", torch.equal(a, b), f"shape {tuple(a.shape)}")
    return a


# ----------------------------------------------------------------------------- 2, 3


def test_cad_alpha0(model, idx, stop, ref):
    b = run(model, idx, 5, max_new_tokens=320, stop_fn=stop, cad=dict(alpha=0.0, short_len=32, beta=0.0), **GEN)
    check("2  CAD alpha=0 (beta=0) == normal decoding", torch.equal(ref, b))
    c = run(model, idx, 5, max_new_tokens=320, stop_fn=stop, cad=dict(alpha=0.0, short_len=32, beta=0.1), **GEN)
    same = sum(torch.equal(x, y) for x, y in zip(ref, c))
    print(f"   info: alpha=0 with the default beta=0.1 filter: {same}/{len(ref)} rows identical "
          f"(the filter alone can remove tokens that top-p 0.9 keeps)")


def test_cad_long_short(model, idx, stop, ref):
    T = idx.size(1)
    b, trace = run(model, idx, 5, max_new_tokens=320, stop_fn=stop,
                   cad=dict(alpha=0.5, short_len=T + 100, beta=0.0), return_logits=True, **GEN)
    eq = all(torch.equal(f, s) for f, s in trace)
    check("3a short_len >= prompt length: logits_short == logits_full at every step", eq, f"{len(trace)} steps")
    check("3b ... and output == normal decoding (alpha=0.5, beta=0)", torch.equal(ref, b))


# ----------------------------------------------------------------------------- 4


def test_short_cache(model, idx, stop, k=32):
    control = {}
    for bf16 in (False, True):      # control: the existing full-prompt cache path, same comparison
        gen, trace = run(model, idx[:4], 7, bf16=bf16, max_new_tokens=160, stop_fn=None,
                         cad=dict(alpha=0.5, short_len=k, beta=0.1), return_logits=True, **GEN)
        seq = torch.cat([idx[:4], gen], dim=1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            full, _ = model(seq)
        T = idx.size(1)
        full = full[:, T - 1:T - 1 + len(trace)].float()
        full[..., :N_SPECIAL] = float("-inf")
        inc = torch.stack([f for f, _ in trace], dim=1)
        fin = torch.isfinite(inc)
        control[bf16] = ((inc[fin] - full[fin]).abs().max().item(),
                         (inc.softmax(-1) - full.softmax(-1)).abs().max().item(),
                         (inc.argmax(-1) == full.argmax(-1)).float().mean().item())
        print(f"   control ({'bf16' if bf16 else 'fp32'}): existing full-cache path vs full forward: "
              f"max |dlogit| {control[bf16][0]:.2e}, max |dp| {control[bf16][1]:.2e}, top-1 agree {control[bf16][2]:.4f}")
    for bf16, atol in [(False, 1e-3), (True, None)]:
        gen, trace = run(model, idx[:4], 7, bf16=bf16, max_new_tokens=160, stop_fn=None,
                         cad=dict(alpha=0.5, short_len=k, beta=0.1), return_logits=True, **GEN)
        seq = torch.cat([idx[:4, -k:], gen], dim=1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            full, _ = model(seq)                                   # training path: FlexAttention sliding mask
        full = full[:, k - 1:k - 1 + len(trace)].float()
        full[..., :N_SPECIAL] = float("-inf")
        inc = torch.stack([s for _, s in trace], dim=1)
        fin = torch.isfinite(inc)
        diff = (inc[fin] - full[fin]).abs()
        top1 = (inc.argmax(-1) == full.argmax(-1)).float().mean().item()
        pdiff = (inc.softmax(-1) - full.softmax(-1)).abs().max().item()
        info = f"max |dlogit| {diff.max():.2e}, mean {diff.mean():.2e}, max |dp| {pdiff:.2e}, top-1 agree {top1:.4f}"
        if not bf16:
            check(f"4a fp32: incremental short-cache logits == full forward (k={k}, {len(trace)} steps, atol {atol})",
                  diff.max().item() < atol, info)
        else:
            print(f"   info: atol 1e-3 on bf16 logits is not reachable (bf16 spacing is 0.06 at |logit| 8-16);"
                  f" {info}")
            check("4b bf16: short-cache error within 1.5x of the existing full-cache path's bf16 error "
                  "(max |dlogit| and max |dp|)",
                  diff.max().item() <= 1.5 * control[True][0] and pdiff <= 1.5 * control[True][1],
                  f"short {diff.max():.2e}/{pdiff:.2e} vs control {control[True][0]:.2e}/{control[True][1]:.2e}")


# ----------------------------------------------------------------------------- 5


def test_boost_toy():
    V = 20
    # row 0: specifics [5,6,7] and [6,9], fact ids {5,6,7,9}; row 1: specific [3,4], fact ids {3,4}
    pb = PromptBoost([[[5, 6, 7], [6, 9]], [[3, 4]]], [[5, 6, 7, 9], [3, 4]], V, b=1.0, b_cont=4.0, device="cpu")
    z = torch.zeros(2, V)
    out = pb(z, torch.zeros(2, 0, dtype=torch.long))
    check("5a start boost: +b exactly on each row's fact ids, nothing else",
          torch.equal(out[0].nonzero().flatten(), torch.tensor([5, 6, 7, 9])) and
          torch.equal(out[1].nonzero().flatten(), torch.tensor([3, 4])) and
          torch.allclose(out[out != 0], torch.ones(6)))

    pb0 = PromptBoost([[[5, 6, 7], [6, 9]], [[3, 4]]], [[], []], V, b=0.0, b_cont=4.0, device="cpu")
    gen = torch.tensor([[1, 2, 5, 6], [8, 8, 8, 3]])              # row 0 ends "5 6", row 1 ends "3"
    out = pb0(torch.zeros(2, V), gen)
    check("5b longest match wins: '5 6' matches [5,6,7] (n=2) over [6,9] (n=1) -> only token 7 gets +b_cont",
          out[0].nonzero().flatten().tolist() == [7] and out[0, 7].item() == 4.0)
    check("5c per-row facts: row 1 ending '3' boosts its own next token 4, not row 0's",
          out[1].nonzero().flatten().tolist() == [4] and out[1, 4].item() == 4.0)
    out = pb0(torch.zeros(2, V), torch.tensor([[1, 2, 3, 6], [5, 6, 7, 1]]))
    check("5d history ending '6': only [6,9] has that prefix -> +b_cont on 9; row 1 (own facts [3,4]) no boost",
          out[0].nonzero().flatten().tolist() == [9] and not out[1].any())
    out = pb0(torch.zeros(2, V), torch.tensor([[5, 6, 7, 8], [1, 1, 1, 4]]))
    check("5e a completed specific does not trigger (no next token): no boost", not out.any())
    pb2 = PromptBoost([[[5, 6, 7], [5, 6, 8], [6, 9]]], [[]], V, b=0.0, b_cont=4.0, device="cpu")
    out = pb2(torch.zeros(1, V), torch.tensor([[5, 6]]))
    check("5f ties at the longest n: both next tokens (7, 8) get +b_cont once, the shorter match (9) none",
          out[0].nonzero().flatten().tolist() == [7, 8] and out[0, 7].item() == 4.0)
    # vectorized matcher == plain-Python longest-match reference on random specifics / histories
    g = torch.Generator().manual_seed(0)
    ok = True
    for _ in range(300):
        B = 3
        seqs = [[torch.randint(0, 6, (int(torch.randint(1, 6, (1,), generator=g)),), generator=g).tolist()
                 for _ in range(int(torch.randint(0, 5, (1,), generator=g)))] for _ in range(B)]
        t = int(torch.randint(0, 9, (1,), generator=g))
        gen = torch.randint(0, 6, (B, t), generator=g)
        got = PromptBoost(seqs, [[]] * B, 8, b=0.0, b_cont=1.0, device="cpu").continuation_targets(gen)
        for r in range(B):
            h, want, best = gen[r].tolist(), set(), 0
            for n in range(1, t + 1):
                nx = {s[n] for s in seqs[r] if len(s) > n and h[t - n:] == s[:n]}
                if nx:
                    want, best = nx, n
            ok &= set(got[r].nonzero().flatten().tolist()) == want
    check("5i vectorized continuation matcher == brute-force longest match (300 random cases)", bool(ok))
    x = torch.tensor([[2.0, 1.0, float("-inf"), 0.5]])
    y = torch.tensor([[1.0, 1.5, float("-inf"), 0.5]])
    c = cad_combine(x, y, alpha=0.5, beta=0.0)
    check("5g cad_combine = (1+a)*full - a*short, -inf stays -inf",
          torch.allclose(c[0, [0, 1, 3]], torch.tensor([2.5, 0.75, 0.5])) and c[0, 2] == float("-inf"))
    c = cad_combine(x, y, alpha=0.5, beta=0.3)
    p = x.softmax(-1)
    check("5h plausibility filter: tokens with p_full < beta * max p_full -> -inf",
          torch.equal(torch.isinf(c), (p < 0.3 * p.max()) | torch.isinf(x)))


# ----------------------------------------------------------------------------- 6


def test_batched_vs_alone(model, idx, stop, facts, ids):
    V = model.cfg.vocab_size
    rows = list(range(6))
    seqs, fids = [facts[ids[r]]["seqs"] for r in rows], [facts[ids[r]]["ids"] for r in rows]
    cad = dict(alpha=0.5, short_len=32, beta=0.1)
    for bf16 in (False, True):
        kw = dict(max_new_tokens=320, stop_fn=stop, cad=cad, temperature=0.0, repetition_penalty=1.1)
        batched = run(model, idx[rows], 0, bf16=bf16, boost=PromptBoost(seqs, fids, V), **kw)
        alone = [run(model, idx[[r]], 0, bf16=bf16, boost=PromptBoost([seqs[r]], [fids[r]], V), **kw)[0] for r in rows]
        same = [torch.equal(batched[i, :len(a)], a) for i, a in enumerate(alone)]
        n = sum(same)
        if not bf16:
            check("6a fp32 greedy, boost+CAD, 6 prompts with their own facts: batched == alone", n == len(rows),
                  f"{n}/{len(rows)} identical")
        else:
            first = [next((j for j in range(len(a)) if batched[i, j] != a[j]), None) for i, a in enumerate(alone)]
            print(f"   info: bf16 greedy: {n}/{len(rows)} identical; first differing step per row: {first} "
                  f"(bf16 matmuls differ by batch shape, so long greedy runs can drift)")
    # sampling with per-row facts: the processor itself is row-independent
    pb_b = PromptBoost(seqs, fids, V)
    g = torch.randint(N_SPECIAL, V, (len(rows), 40), device="cuda")
    lg = torch.randn(len(rows), V, device="cuda")
    ok = all(torch.equal(pb_b(lg, g)[i], PromptBoost([seqs[i]], [fids[i]], V)(lg[[i]], g[[i]])[0]) for i in rows)
    check("6b PromptBoost on a batch == PromptBoost on each row alone (random logits/history)", ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(PROJECT_DIR, "runs", "archD_4B_docmask", "ckpt_best.pt"))
    ap.add_argument("--max-mem-gb", type=float, default=3.0)
    ap.add_argument("--skip-subprocess", action="store_true")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.max_mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)

    test_boost_toy()
    from transformers import PreTrainedTokenizerFast
    prompts = json.load(open(PROMPTS))
    ids = random.Random(0).sample(sorted(prompts), 20)
    if not a.skip_subprocess:
        test_generate_py_unchanged(a.ckpt, ids, prompts)
    model, cfg = load_model(a.ckpt)
    patch_head(model)
    tok = PreTrainedTokenizerFast.from_pretrained(os.path.join(PROJECT_DIR, "artifacts", "tokenizer",
                                                               cfg["data"]["tokenizer"]))
    starts = word_start_table(tok).cuda()
    stop = lambda g: starts[g].sum(1) > TARGET_WORDS
    idx = batch_of(tok, prompts, ids, model.cfg.max_seq_len - 320)
    print(f"20 prompts, common length {idx.size(1)} tokens")
    ref = test_loop_equivalence(model, idx, stop)
    test_cad_alpha0(model, idx, stop, ref)
    test_cad_long_short(model, idx, stop, ref)
    test_short_cache(model, idx, stop)
    test_batched_vs_alone(model, idx, stop, json.load(open(FACTS)), ids)
    print(f"\npeak CUDA memory allocated by the tests: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    print(f"{sum(ok for _, ok in results)}/{len(results)} checks passed")
    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == "__main__":
    main()
