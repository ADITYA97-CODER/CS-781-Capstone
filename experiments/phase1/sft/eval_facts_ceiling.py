"""Ceiling test: does giving SFT-E the right facts (a plain-text block before the prompt) improve its continuations?

  python phase1/sft/eval_facts_ceiling.py --stage blocks                 # held-out facts blocks + their stats
  python phase1/sft/eval_facts_ceiling.py --stage single --gpus 0,1      # seed 1 for every config + REPORT_single.md
  python phase1/sft/eval_facts_ceiling.py --stage gen --gpus 0,1         # seeds 1-8 + penalty-aware MBR
  python phase1/sft/eval_facts_ceiling.py --stage report                 # REPORT.md (MBR x8)

Configs (decoding identical for all: the current best pipeline without CAD = generate.py T 0.7 / top-p 0.9 / rep 1.1,
header + citation steering, 8 seeds, phase1/mbr/mbr_rerank.py with the frozen archE_local_rerank8_tuning weights):
  1 start    runs/sft_archE_v2_upweight (no facts)  -> EXISTING samples outputs/phase1/masked_sft/eval/sft_upweight (same flags)
  2 C        runs/sft_facts_C/ckpt_final.pt (no facts; both runs compared at their final step 100 = same steps)
  3 F_oracle runs/sft_facts_F + prompt specifics used by the reference      (ceiling)
  4 F_all    runs/sft_facts_F + all prompt specifics (block capped at 150 tokens, oldest dropped)
  5 F_empty  runs/sft_facts_F + "FACTS: END FACTS"
  6 F_random runs/sft_facts_F + the oracle block of a random other held-out prompt (fixed-seed, donor's block non-empty)
Blocks have no noise at test time. Δ = paired difference vs C, 95% bootstrap CI over the 990 prompts.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJECT_DIR, "phase1", "diagnostics"))
from eval_masked_sft import GEN, JUNK, PY, SEEDS, WEIGHTS, run_jobs  # noqa: E402
from facts import block_ids, block_text, key_in, mark_used, prompt_facts  # noqa: E402

P = lambda *a: os.path.join(PROJECT_DIR, *a)
OUT = P("outputs", "phase1", "facts_ceiling")
BLOCKS = os.path.join(OUT, "blocks")
START_DIR = P("outputs", "phase1", "masked_sft", "eval")          # existing sft_upweight samples
CONFIGS = {  # name -> (checkpoint, facts kind or None)
    "start": ("runs/sft_archE_v2_upweight/ckpt_best.pt", None),
    "C": ("runs/sft_facts_C/ckpt_final.pt", None),
    "F_oracle": ("runs/sft_facts_F/ckpt_final.pt", "oracle"),
    "F_all": ("runs/sft_facts_F/ckpt_final.pt", "all"),
    "F_empty": ("runs/sft_facts_F/ckpt_final.pt", "empty"),
    "F_random": ("runs/sft_facts_F/ckpt_final.pt", "random"),
}
LABEL = {"start": "(1) start SFT-E, no facts", "C": "(2) Run C, no facts", "F_oracle": "(3) F + oracle facts",
         "F_all": "(4) F + all prompt facts", "F_empty": "(5) F + empty block", "F_random": "(6) F + random other prompt's facts"}


def sample_dir(name, seed):
    if name == "start":
        return os.path.join(START_DIR, "sft_upweight", f"s{seed}")
    return os.path.join(OUT, name, f"s{seed}")


def mbr_path(name):
    return os.path.join(START_DIR, "sft_upweight_mbr8.json") if name == "start" else os.path.join(OUT, f"{name}_mbr8.json")


# ----------------------------------------------------------------------------- blocks

def stage_blocks(a=None):
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(P("artifacts", "tokenizer", "legal_bpe_32k"))
    prompts = json.load(open(P("data", "corpus", "eval", "local_test.json")))
    refs = json.load(open(P("data", "corpus", "eval", "local_ref.json")))
    ids = list(prompts)
    cache = P("outputs", "phase1", "fact_decoding", "ner_prompts_refs.json")      # spaCy NER keys (Ctx.run_ner)
    ner = json.load(open(cache))["prompt"]
    os.makedirs(BLOCKS, exist_ok=True)
    facts = {k: mark_used(prompt_facts(prompts[k]["text"], ner[k]), refs[k]["text"]) for k in ids}
    chosen = {"oracle": {k: [f for f in facts[k] if f["used"]] for k in ids}, "all": facts, "empty": {k: [] for k in ids}}
    rng = np.random.default_rng(0)
    donors = [k for k in ids if chosen["oracle"][k]]
    rand, donor_of = {}, {}
    for k in ids:
        d = k
        while d == k:
            d = donors[int(rng.integers(len(donors)))]
        donor_of[k] = d
        rand[k] = chosen["oracle"][d]
    chosen["random"] = rand
    stats = {}
    for kind, per in chosen.items():
        out = {}
        for k in ids:
            ids_, kept = block_ids(tok, per[k])
            out[k] = {"text": block_text(kept), "facts": kept, "n_tokens": len(ids_), "capped": len(per[k]) - len(kept)}
            if kind == "random":
                out[k]["donor"] = donor_of[k]
        json.dump(out, open(os.path.join(BLOCKS, f"{kind}.json"), "w"), indent=1, ensure_ascii=False)
        stats[kind] = dict(facts_per_block=float(np.mean([len(v["facts"]) for v in out.values()])),
                           block_tokens_mean=float(np.mean([v["n_tokens"] for v in out.values()])),
                           block_tokens_max=int(max(v["n_tokens"] for v in out.values())),
                           empty_share=float(np.mean([not v["facts"] for v in out.values()])),
                           capped_blocks=int(sum(v["capped"] > 0 for v in out.values())))
    # random facts that happen to be true for the receiving prompt (in its prompt or reference)
    pr = {k: (prompts[k]["text"], refs[k]["text"]) for k in ids}
    from attention_reach import regex_specifics
    hit = 0
    tot = 0
    for k in ids:
        prk, rrk = set(regex_specifics(pr[k][0])), set(regex_specifics(pr[k][1]))
        for f in rand[k]:
            tot += 1
            hit += any(key_in(x, pr[k][0], prk) or key_in(x, pr[k][1], rrk) for x in f["keys"])
    stats["random"]["facts_true_for_receiver"] = hit / max(1, tot)
    json.dump(stats, open(os.path.join(BLOCKS, "stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))
    return stats


# ----------------------------------------------------------------------------- generation

def gen_jobs(names, seeds):
    jobs = []
    for n in names:
        if n == "start":
            continue                                  # existing samples
        ckpt, kind = CONFIGS[n]
        if not os.path.exists(P(ckpt)):
            sys.exit(f"missing checkpoint for {n}: {ckpt}")
        for s in seeds:
            d = sample_dir(n, s)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            extra = ["--prefix-facts", os.path.join(BLOCKS, f"{kind}.json")] if kind else []
            jobs.append(([PY, P("phase1/generate.py"), "--ckpt", P(ckpt), "--prompts", "local", *GEN, *extra,
                          "--seed", str(s), "--out", d], d + ".log", os.path.join(d, "prediction.json")))
    return jobs


def stage_gen(a, single=False):
    import subprocess
    if not os.path.exists(os.path.join(BLOCKS, "stats.json")):
        stage_blocks()
    gpus = [int(g) for g in a.gpus.split(",")]
    names = a.configs.split(",") if a.configs else list(CONFIGS)
    seeds = [1] if single else SEEDS
    for s in seeds:
        if not os.path.exists(os.path.join(sample_dir("start", s), "prediction.json")):
            sys.exit(f"missing existing start samples {sample_dir('start', s)}")
    run_jobs(gen_jobs(names, seeds), gpus)
    if single:
        return
    w = json.load(open(WEIGHTS))["weights"]
    for n in names:
        out = mbr_path(n)
        if os.path.exists(out):
            continue
        subprocess.run([PY, P("phase1/mbr/mbr_rerank.py"), "--cands",
                        *[os.path.join(sample_dir(n, s), "prediction.json") for s in SEEDS],
                        "--prompts", "local", "--weights", json.dumps(w), "--out", out, "--batch-size", "64"],
                       check=True, cwd=PROJECT_DIR, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpus[0])},
                       stdout=subprocess.DEVNULL)
        print("rerank", n, out, flush=True)


def stage_single(a):
    stage_gen(a, single=True)
    stage_report(a, mode="single")


# ----------------------------------------------------------------------------- report

def stage_report(a, mode="mbr8"):
    from attention_reach import regex_specifics
    from eval_fact_decoding import CLOSERS, HEADERS, SOURCES, Ctx, compute_metrics, ngrams, paired_diff_ci, self_rep
    sys.path.insert(0, P("phase1", "mbr"))
    from mbr_rerank import SIGN
    single = mode == "single"
    seeds = [1] if single else SEEDS
    pred_path = (lambda n: os.path.join(sample_dir(n, 1), "prediction.json")) if single else mbr_path
    names = [n for n in CONFIGS if os.path.exists(pred_path(n))]
    if "C" not in names:
        sys.exit("Run C outputs missing")
    ctx = Ctx()
    ids = list(ctx.refs)
    refs = [ctx.refs[k]["text"] for k in ids]
    src = np.array([ctx.refs[k]["source"] for k in ids])
    blocks = {kind: json.load(open(os.path.join(BLOCKS, f"{kind}.json"))) for kind in ("oracle", "all", "empty", "random")}
    bstats = json.load(open(os.path.join(BLOCKS, "stats.json")))
    nonempty = np.array([bool(blocks["oracle"][k]["facts"]) for k in ids])

    res, txts = {}, {}
    for n in names:
        res[n] = compute_metrics(ctx, pred_path(n), ids, [sample_dir(n, s) for s in seeds])
        pred = json.load(open(pred_path(n)))
        txts[n] = [pred[k]["text"] for k in ids]
        print("metrics", n, f"{res[n]['bertscore']:.4f}", flush=True)

    def windows(txt):
        cand, ref = [], []
        for a_ in range(0, 128, 32):
            cand += [" ".join(t.split()[a_:a_ + 32]) or "." for t in txt]
            ref += [" ".join(r.split()[a_:a_ + 32]) or "." for r in refs]
        _, _, f = ctx.scorer.score(cand, ref)
        return np.array(f.tolist()).reshape(4, len(ids))

    def extras(txt, is_ref=False):
        inv_cit = inv_date = n_cit = n_date = junk = sign = 0
        for i, k in enumerate(ids):
            t = txt[i]
            rx = regex_specifics(t)
            p_keys, r_keys = ctx.regex[("prompt", k)], ctx.regex[("ref", k)]
            for key, (typ, _) in rx.items():
                new = key not in p_keys and (is_ref or key not in r_keys)
                if typ == "citation":
                    n_cit += 1
                    inv_cit += new
                if typ == "date":
                    n_date += 1
                    inv_date += new
            junk += bool(JUNK.search(t))
            sign += bool(SIGN.search(t))
        return dict(citations=n_cit, invented_citations=inv_cit, dates=n_date, invented_dates=inv_date, junk=junk,
                    signature_blocks=sign)

    def facts_used(txt, kind):
        used = tot = 0
        for i, k in enumerate(ids):
            rk = set(regex_specifics(txt[i]))
            for f in blocks[kind][k]["facts"]:
                tot += 1
                used += any(key_in(x, txt[i], rk) for x in f["keys"])
        return used / tot if tot else float("nan"), tot

    win = {n: windows(txts[n]) for n in names}
    for n in names:
        r = res[n]
        r.update(extras(txts[n]))
        kind = CONFIGS[n][1]
        # facts-used rate: for no-facts configs, the share of the ALL / ORACLE block facts that appear anyway
        r["facts_used"], r["facts_in_blocks"] = facts_used(txts[n], kind or "all")
        r["oracle_facts_used"], _ = facts_used(txts[n], "oracle")
        vis = os.path.join(sample_dir(n, 1), "generation_stats.json")
        r["visible_prompt_tokens"] = json.load(open(vis)).get("visible_prompt_tokens_mean") if os.path.exists(vis) else None
    ref_cp = [np.mean([x in ctx.p8[k] for x in ngrams(refs[i].split(), 8)]) for i, k in enumerate(ids)]
    ref_rates = dict(closing=sum(bool(CLOSERS.search(t)) for t in refs), loops=sum(self_rep(t.split()) > 0.25 for t in refs),
                     copy8=float(np.mean(ref_cp)), **extras(refs, is_ref=True))
    ref_rates["oracle_facts_used"] = facts_used(refs, "oracle")[0]
    ref_rates["all_facts_used"] = facts_used(refs, "all")[0]
    bs = {n: np.array(res[n]["bertscore_per_prompt"]) for n in names}
    base = "C"
    for n in names:
        if n == base:
            continue
        r = res[n]
        r["bs_diff"], r["bs_diff_ci"] = paired_diff_ci(bs[n], bs[base])
        r["bs_diff_source"] = {s: paired_diff_ci(bs[n][src == s], bs[base][src == s]) for s in SOURCES}
        r["bs_diff_window"] = {f"{i * 32 + 1}-{i * 32 + 32}": paired_diff_ci(win[n][i], win[base][i]) for i in range(4)}
        r["bs_diff_nonempty_oracle"] = paired_diff_ci(bs[n][nonempty], bs[base][nonempty])
        r["bs_diff_empty_oracle"] = paired_diff_ci(bs[n][~nonempty], bs[base][~nonempty])
    dec = {}
    pairs = [("ceiling (3)-(2)", "F_oracle", "C"), ("achievable without selector (4)-(2)", "F_all", "C"),
             ("gap a perfect selector could close (3)-(4)", "F_oracle", "F_all"), ("sanity (6)-(5)", "F_random", "F_empty"),
             ("block format alone (5)-(2)", "F_empty", "C"), ("Run C vs start (2)-(1)", "C", "start")]
    for lab, x, y in pairs:
        if x in bs and y in bs:
            dec[lab] = dict(all=paired_diff_ci(bs[x], bs[y]), nonempty_oracle=paired_diff_ci(bs[x][nonempty], bs[y][nonempty]))

    L = report_md(res, names, ref_rates, dec, bstats, nonempty, single)
    suffix = "_single" if single else ""
    open(os.path.join(OUT, f"REPORT{suffix}.md"), "w").write("\n".join(L) + "\n")
    for r in res.values():
        for k in ("bertscore_per_prompt", "loops_per_prompt", "copy8_per_prompt", "reuse_hits_tots"):
            r.pop(k, None)
    json.dump(dict(results=res, reference_rates=ref_rates, decision=dec, block_stats=bstats),
              open(os.path.join(OUT, f"metrics{suffix}.json"), "w"), indent=1, default=float)
    print("\n".join(L))


def report_md(res, names, ref, dec, bstats, nonempty, single):
    f4 = lambda v: f"{v:+.4f}"
    ci = lambda d, c: f"{f4(d)} ({f4(c[0])}…{f4(c[1])})"
    excl = lambda c: " ✓" if c[0] > 0 else (" ✗" if c[1] < 0 else "")
    how = ("ONE sample per prompt (seed 1; T 0.7, top-p 0.9, rep 1.1, header + citation steering), no MBR" if single else
           "8 samples (T 0.7, top-p 0.9, rep 1.1, header + citation steering) + penalty-aware MBR (frozen weights)")
    L = [f"# Facts-block ceiling test — 990 local prompts ({'single sample, seed 1' if single else 'MBR x8'})", "",
         f"Decoding for every config: {how}. No CAD. Δ = paired difference vs **Run C** (same pairs, steps, LR and data "
         "order, no block), 95% bootstrap CI over prompts. (1) reuses the existing SFT-upweight samples (same flags).", "",
         "## Decision summary", ""]
    for lab, d in dec.items():
        L.append(f"- **{lab}**: {ci(*d['all'])}{excl(d['all'][1])}  | on the {int(nonempty.sum())} prompts with a non-empty "
                 f"oracle block: {ci(*d['nonempty_oracle'])}")
    if "sanity (6)-(5)" in dec:
        d = dec["sanity (6)-(5)"]["all"]
        L.append("")
        L.append("**SANITY FLAG: (6) > (5) — the model may be copying facts blindly.**" if d[0] > 0 else
                 "Sanity OK: (6) ≤ (5) (wrong facts do not help).")
    L += ["", "## Facts blocks (held-out)", "",
          "| block | facts / block | tokens / block (mean, max) | empty blocks | capped at 150 |", "|---|---|---|---|---|"]
    for kind in ("oracle", "all", "empty", "random"):
        b = bstats[kind]
        L.append(f"| {kind} | {b['facts_per_block']:.2f} | {b['block_tokens_mean']:.1f}, {b['block_tokens_max']} | "
                 f"{b['empty_share']:.3f} | {b['capped_blocks']} |")
    L.append(f"\nShare of held-out prompts with an EMPTY oracle block: **{bstats['oracle']['empty_share']:.3f}**. "
             f"Random-block facts that happen to be true for the receiving prompt: {bstats['random']['facts_true_for_receiver']:.3f}.")
    L += ["", "## Main table", "",
          "| config | BERTScore | Δ vs C (95% CI) | ROUGE-L | reuse | invented rate | invented / output | invented citations | invented dates | facts used | oracle facts used |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        r = res[n]
        d = "—" if n == "C" else ci(r["bs_diff"], r["bs_diff_ci"])
        fu = f"{r['facts_used']:.3f}" if CONFIGS[n][1] not in (None, "empty") else ("—" if CONFIGS[n][1] == "empty" else f"({r['facts_used']:.3f} of all)")
        L.append(f"| {LABEL[n]} | {r['bertscore']:.4f} | {d} | {r['rouge_l']:.4f} | {r['reuse_overall']:.3f} | "
                 f"{r['invented_rate']:.3f} | {r['invented_per_output']:.2f} | {r['invented_citations']}/{r['citations']} | "
                 f"{r['invented_dates']}/{r['dates']} | {fu} | {r['oracle_facts_used']:.3f} |")
    L.append(f"| references | — | — | — | — | — | — | (new vs prompt) {ref['invented_citations']}/{ref['citations']} | "
             f"{ref['invented_dates']}/{ref['dates']} | ({ref['all_facts_used']:.3f} of all) | {ref['oracle_facts_used']:.3f} |")
    L += ["", "facts used = share of the facts in the block that appear in the output (no-facts configs: share of the "
          "ALL-block facts that appear anyway). oracle facts used = same for the oracle block (references: 1.0 by definition).", "",
          "| config | loops | 8-gram prompt copy | closing | signature blocks | junk | outputs < 128 words | samples < 128 words | visible prompt tokens (seed 1) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        r = res[n]
        v = f"{r['visible_prompt_tokens']:.1f}" if r.get("visible_prompt_tokens") else "standard"
        L.append(f"| {LABEL[n]} | {r['loops']} | {r['copy8']:.4f} | {r['closing']} | {r['signature_blocks']} | {r['junk']} | "
                 f"{r['short']} | {r['short_samples']} | {v} |")
    L.append(f"| references | {ref['loops']} | {ref['copy8']:.4f} | {ref['closing']} | — | {ref['junk']} | — | — | — |")
    L += ["", "\"ended samples\" = outputs / samples shorter than 128 words (generation ran out of budget, e.g. junk).", "",
          "## BERTScore Δ vs Run C per source and per 32-word window (95% CI)", "",
          "| config | Supreme Court | High Courts | Tribunals | Acts | w 1-32 | w 33-64 | w 65-96 | w 97-128 | non-empty oracle | empty oracle |",
          "|---|" + "---|" * 10]
    for n in names:
        if n == "C":
            continue
        r = res[n]
        cells = [ci(*r["bs_diff_source"][s]) for s in ("supreme_court", "high_courts", "tribunals", "acts")]
        cells += [ci(*r["bs_diff_window"][w]) for w in ("1-32", "33-64", "65-96", "97-128")]
        cells += [ci(*r["bs_diff_nonempty_oracle"]), ci(*r["bs_diff_empty_oracle"])]
        L.append(f"| {LABEL[n]} | " + " | ".join(cells) + " |")
    L += ["", "Absolute BERTScore per source / window:", ""]
    for n in names:
        r = res[n]
        L.append(f"- {LABEL[n]}: " + ", ".join(f"{k} {v:.4f}" for k, v in r["bertscore_source"].items()) + " | "
                 + ", ".join(f"{k} {v:.4f}" for k, v in r["bertscore_windows"].items()))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["blocks", "single", "gen", "report"])
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--configs", default=None, help="comma list (default all)")
    a = ap.parse_args()
    {"blocks": stage_blocks, "single": stage_single, "gen": stage_gen, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
