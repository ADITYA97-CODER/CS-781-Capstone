"""Evaluate prompt-token boost and context-aware decoding (generate.py --prompt-boost / --cad) on Arch D.

Every config = 4 samples (seeds 1-4, T 0.7, top-p 0.9, repetition penalty 1.1, batch 128, 320 new
tokens, prompts truncated exactly as in the 990-prompt batch-128 run) + BERTScore MBR (mbr_select.py).

  CUDA_VISIBLE_DEVICES=1 python phase1/diagnostics/eval_fact_decoding.py --stage A     # 300-prompt subset, 10 configs
  CUDA_VISIBLE_DEVICES=1 python phase1/diagnostics/eval_fact_decoding.py --stage B     # 990 prompts, baseline + top 3
  CUDA_VISIBLE_DEVICES=1 python phase1/diagnostics/eval_fact_decoding.py --stage official --config <name>  # 8 samples + MBR
  python phase1/diagnostics/eval_fact_decoding.py --stage report

Everything is resumable (existing samples / MBR files / metrics are reused).
Output: outputs/phase1/fact_decoding/{A,B,official}/<config>/..., REPORT.md, plots.

Metrics per config (MBR output vs local references):
  BERTScore F1 overall / per source / per 32-word window, ROUGE-L, paired-bootstrap CI of the
  BERTScore difference to baseline; reuse rate of prompt specifics by distance (Phase-1 buckets, cluster
  bootstrap CIs, attention_reach.py definitions); invented-specifics rate (output specifics in neither
  prompt nor reference); closing-formula and court-header counts (output_formulas.py regexes); loops
  (outputs with > 25% repeated 4-grams, error_analysis.py definition); verbatim copying (share of output
  word 8-grams that occur in the prompt); generation time per prompt (4 samples).
"""

import argparse
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from attention_reach import (BUCKET_NAMES, bucket_of, cluster_bootstrap, ner_norm, ner_ok,  # noqa: E402
                             ner_pattern, regex_specifics, visible_lengths)

PROJECT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
P = lambda *a: os.path.join(PROJECT_DIR, *a)
OUT = P("outputs", "phase1", "fact_decoding")
CKPT = P("runs", "archD_4B_docmask", "ckpt_best.pt")
SEEDS = [1, 2, 3, 4]
GEN_ARGS = ["--temperature", "0.7", "--top-p", "0.9", "--repetition-penalty", "1.1", "--batch-size", "128",
            "--trunc-group", "128", "--max-new-tokens", "320", "--max-mem-gb", "20"]
SOURCES = ["supreme_court", "high_courts", "tribunals", "acts"]
HEADERS = re.compile(r"IN THE (SUPREME|HIGH) COURT|\bREPORTABLE\b|(CIVIL|CRIMINAL) (APPELLATE|ORIGINAL) JURISDICTION")
CLOSERS = re.compile(r"\b(appeal is (dis|al)lowed|petition is (dis|al)lowed|appeal dismissed|appeal allowed|"
                     r"no order as to costs|rule (is )?(made absolute|discharged)|parties to bear|"
                     r"stands? disposed of|bail bonds)", re.I)


def boost(b, c=4.0):
    return ["--prompt-boost", "--boost-b", str(b), "--boost-cont", str(c)]


def cad(a, k=32, beta=0.1):
    return ["--cad", "--cad-alpha", str(a), "--cad-short-len", str(k), "--cad-beta", str(beta)]


STAGE_A = {
    "baseline": [],
    "boost_b0.5": boost(0.5), "boost_b1.0": boost(1.0), "boost_b2.0": boost(2.0),
    "cad_a0.3": cad(0.3), "cad_a0.5": cad(0.5), "cad_a1.0": cad(1.0),
    "cad_a0.5_k16": cad(0.5, 16), "cad_a0.5_k64": cad(0.5, 64),
}


def sh(cmd, log):
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as fh:
        subprocess.run(cmd, check=True, cwd=PROJECT_DIR, stdout=fh, stderr=subprocess.STDOUT)


def run_config(stage_dir, name, flags, prompts="local", subset=None, seeds=SEEDS):
    d = os.path.join(stage_dir, name)
    cands = []
    for s in seeds:
        o = os.path.join(d, f"s{s}")
        if not os.path.exists(os.path.join(o, "prediction.json")):
            t = time.time()
            sh([sys.executable, P("phase1", "generate.py"), "--ckpt", CKPT, "--prompts", prompts, *GEN_ARGS,
                *flags, "--seed", str(s), "--out", o] + (["--subset", subset] if subset else []), o + ".log")
            print(f"  {name} seed {s}: {time.time() - t:.0f}s", flush=True)
        cands.append(os.path.join(o, "prediction.json"))
    mbr = os.path.join(d, "mbr.json")
    if not os.path.exists(mbr):
        sh([sys.executable, P("phase1", "mbr", "mbr_select.py"), "--cands", *cands, "--out", mbr,
            "--batch-size", "64"], os.path.join(d, "mbr.log"))
    json.dump({"name": name, "flags": flags, "ckpt": CKPT, "seeds": seeds, "gen_args": GEN_ARGS},
              open(os.path.join(d, "config.json"), "w"), indent=1)
    return mbr


# ----------------------------------------------------------------------------- metrics


class Ctx:
    """Prompts, references, NER of both, visible prompt geometry and the scorer, shared by all configs."""

    def __init__(self):
        from transformers import PreTrainedTokenizerFast
        import spacy
        import bert_score
        self.tok = PreTrainedTokenizerFast.from_pretrained(P("artifacts", "tokenizer", "legal_bpe_32k"))
        self.prompts = json.load(open(P("data", "corpus", "eval", "local_test.json")))
        self.refs = json.load(open(P("data", "corpus", "eval", "local_ref.json")))
        ids = list(self.prompts)
        enc = {k: self.tok(self.prompts[k]["text"], add_special_tokens=False, return_offsets_mapping=True) for k in ids}
        self.n_tok = {k: len(enc[k]["input_ids"]) for k in ids}
        self.tok_starts = {k: [o[0] for o in enc[k]["offset_mapping"]] for k in ids}
        self.vis_len = visible_lengths(self.prompts, self.tok)          # batch-128 truncation, = our runs
        self.vis_char = {k: (self.tok_starts[k][self.n_tok[k] - self.vis_len[k]]
                             if self.vis_len[k] < self.n_tok[k] else 0) for k in ids}
        self.nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        self.nlp.max_length = 2_000_000
        cache = os.path.join(OUT, "ner_prompts_refs.json")
        if os.path.exists(cache):
            self.ner = json.load(open(cache))
        else:
            self.ner = {"prompt": self.run_ner([self.prompts[k]["text"] for k in ids], ids),
                        "ref": self.run_ner([self.refs[k]["text"] for k in ids], ids)}
            json.dump(self.ner, open(cache, "w"))
        self.regex = {("prompt", k): regex_specifics(self.prompts[k]["text"]) for k in ids}
        self.regex.update({("ref", k): regex_specifics(self.refs[k]["text"]) for k in ids})
        self.scorer = bert_score.BERTScorer(lang="en", batch_size=64, device="cuda")
        self.p8 = {k: set(ngrams(self.prompts[k]["text"].split(), 8)) for k in ids}

    def run_ner(self, texts, ids):
        out = {}
        for k, doc in zip(ids, self.nlp.pipe(texts, batch_size=32, n_process=8)):
            out[k] = sorted({ner_norm(e.text) for e in doc.ents if ner_ok(e)})
        return out

    def specifics(self, text, ner_keys, regex=None):
        sp = dict(regex if regex is not None else regex_specifics(text))
        for nk in ner_keys:
            pos = [m.start() for m in ner_pattern(nk).finditer(text)]
            if pos:
                sp["ner:" + nk] = ("ner", pos)
        return sp

    def occurs(self, key, src, k, text=None):
        if key.startswith("ner:"):
            t = text if text is not None else (self.prompts if src == "prompt" else self.refs)[k]["text"]
            return [m.start() for m in ner_pattern(key[4:]).finditer(t)]
        return self.regex[(src, k)].get(key, (None, []))[1]


def ngrams(w, n):
    return [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]


def self_rep(w, n=4):
    g = Counter(ngrams(w, n))
    tot = sum(g.values())
    return (tot - len(g)) / tot if tot else 0


def boot_mean_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5]).tolist()


def compute_metrics(ctx, pred_path, ids, gen_dirs):
    pred = json.load(open(pred_path))
    txt = [pred[k]["text"] for k in ids]
    refs = [ctx.refs[k]["text"] for k in ids]
    m = {"n": len(ids)}
    # < 128 words happens when a sample ends in space-less junk (e.g. "/tb/tb/..." after a signature) that
    # uses up the 320-token budget; MBR prefers 128-word candidates, so this counts prompts where all 4 failed
    m["short"] = sum(len(t.split()) < 128 for t in txt)
    m["short_samples"] = sum(len(json.load(open(os.path.join(g, "prediction.json")))[k]["text"].split()) < 128
                             for g in gen_dirs for k in ids)

    # BERTScore: whole text + four 32-word windows, one scorer call
    cand, ref = list(txt), list(refs)
    for a in range(0, 128, 32):
        cand += [" ".join(t.split()[a:a + 32]) or "." for t in txt]
        ref += [" ".join(r.split()[a:a + 32]) or "." for r in refs]
    _, _, f = ctx.scorer.score(cand, ref)
    f = np.array(f.tolist()).reshape(5, len(ids))
    m["bertscore_per_prompt"] = f[0].tolist()
    m["bertscore"] = float(f[0].mean())
    m["bertscore_ci"] = boot_mean_ci(f[0])
    m["bertscore_windows"] = {f"{a + 1}-{a + 32}": float(f[1 + a // 32].mean()) for a in range(0, 128, 32)}
    m["bertscore_source"] = {s: float(np.mean([f[0][i] for i, k in enumerate(ids) if ctx.refs[k]["source"] == s]))
                             for s in SOURCES}
    spec = importlib.util.spec_from_file_location("ev", P("official_eval", "Evaluation Script", "evaluate_phase1.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    m["rouge_l"] = float(np.mean([ev.rouge_l_f1(t, r) for t, r in zip(txt, refs)]))

    # specifics
    out_ner = ctx.run_ner(txt, ids)
    hits = np.zeros((len(ids), len(BUCKET_NAMES)))
    tots = np.zeros_like(hits)
    n_out_sp = n_inv = 0
    inv_per = []
    for i, k in enumerate(ids):
        t = txt[i]
        ref_sp = ctx.specifics(refs[i], ctx.ner["ref"][k], ctx.regex[("ref", k)])
        out_sp = ctx.specifics(t, out_ner[k])
        out_regex = regex_specifics(t)
        for key in ref_sp:
            pp = ctx.occurs(key, "prompt", k)
            if not pp or max(pp) < ctx.vis_char[k]:
                continue
            j = bucket_of(ctx.n_tok[k] - (np.searchsorted(ctx.tok_starts[k], max(pp), side="right") - 1))
            tots[i, j] += 1
            hits[i, j] += bool(ner_pattern(key[4:]).search(t)) if key.startswith("ner:") else key in out_regex
        inv = [key for key in out_sp if not ctx.occurs(key, "prompt", k) and not ctx.occurs(key, "ref", k)]
        n_out_sp += len(out_sp)
        n_inv += len(inv)
        inv_per.append((len(inv), len(out_sp)))
    lo, hi = cluster_bootstrap(hits, tots)
    m["reuse"] = dict(n=tots.sum(0).astype(int).tolist(), hits=hits.sum(0).astype(int).tolist(),
                      rate=(hits.sum(0) / np.maximum(tots.sum(0), 1)).tolist(), lo=lo.tolist(), hi=hi.tolist())
    m["reuse_overall"] = float(hits.sum() / max(tots.sum(), 1))
    m["reuse_hits_tots"] = [hits.tolist(), tots.tolist()]
    iv = np.array(inv_per, dtype=float)
    bi = np.random.default_rng(0).integers(0, len(iv), size=(2000, len(iv)))
    m["invented_rate"] = float(n_inv / max(n_out_sp, 1))
    m["invented_ci"] = np.percentile(iv[bi, 0].sum(1) / np.maximum(iv[bi, 1].sum(1), 1), [2.5, 97.5]).tolist()
    m["specifics_per_output"] = float(n_out_sp / len(ids))
    m["invented_per_output"] = float(n_inv / len(ids))

    # formulas, loops, copying
    m["closing"] = sum(bool(CLOSERS.search(t)) for t in txt)
    m["header"] = sum(bool(HEADERS.search(t)) for t in txt)
    rep = [self_rep(t.split()) for t in txt]
    m["loops_per_prompt"] = [float(r > 0.25) for r in rep]
    m["loops"] = int(sum(m["loops_per_prompt"]))
    cp = []
    for i, k in enumerate(ids):
        g = ngrams(txt[i].split(), 8)
        cp.append(sum(x in ctx.p8[k] for x in g) / len(g))
    m["copy8_per_prompt"] = cp
    m["copy8"] = float(np.mean(cp))
    m["copy8_ci"] = boot_mean_ci(cp)

    # time: generation seconds of the 4 samples / prompts, and peak memory
    st = [json.load(open(os.path.join(g, "generation_stats.json"))) for g in gen_dirs
          if os.path.exists(os.path.join(g, "generation_stats.json"))]       # absent for pre-change runs
    m["sec_per_prompt_4samples"] = float(sum(s["seconds"] for s in st) / st[0]["n_prompts"]) if st else float("nan")
    m["peak_alloc_gb"] = float(max(s["peak_alloc_gb"] for s in st)) if st else float("nan")
    return m


def paired_diff_ci(a, b, n_boot=2000, seed=0):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    return float(d.mean()), np.percentile(d[idx].mean(1), [2.5, 97.5]).tolist()


def evaluate_stage(stage, configs, ids, subset=None):
    stage_dir = os.path.join(OUT, stage)
    os.makedirs(stage_dir, exist_ok=True)
    for name, flags in configs.items():
        print(f"[{stage}] {name}", flush=True)
        run_config(stage_dir, name, flags, subset=subset)
    ctx = Ctx()
    res_path = os.path.join(stage_dir, "metrics.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for name, flags in configs.items():
        if name not in res or res[name].get("flags") != flags:
            d = os.path.join(stage_dir, name)
            res[name] = compute_metrics(ctx, os.path.join(d, "mbr.json"), ids, [os.path.join(d, f"s{s}") for s in SEEDS])
            res[name]["flags"] = flags
            json.dump(res, open(res_path, "w"))
            print(f"  {name}: BERTScore {res[name]['bertscore']:.4f} loops {res[name]['loops']} "
                  f"copy8 {res[name]['copy8']:.3f} reuse {res[name]['reuse_overall']:.3f}", flush=True)
    base = res["baseline"]
    for name, r in res.items():
        r["bs_diff"], r["bs_diff_ci"] = paired_diff_ci(r["bertscore_per_prompt"], base["bertscore_per_prompt"])
        r["loop_diff"], r["loop_diff_ci"] = paired_diff_ci(r["loops_per_prompt"], base["loops_per_prompt"])
        r["copy_diff"], r["copy_diff_ci"] = paired_diff_ci(r["copy8_per_prompt"], base["copy8_per_prompt"])
        # "clearly above baseline": paired 95% CI entirely above 0
        r["flag_loops"] = bool(r["loop_diff_ci"][0] > 0)
        r["flag_copy"] = bool(r["copy_diff_ci"][0] > 0)
    json.dump(res, open(res_path, "w"))
    return res


def subset_ids(n=300, seed=0):
    path = os.path.join(OUT, "subset300.json")
    if os.path.exists(path):
        return json.load(open(path))
    refs = json.load(open(P("data", "corpus", "eval", "local_ref.json")))
    by = defaultdict(list)
    for k, v in refs.items():
        by[v["source"]].append(k)
    quota = {s: round(n * len(by[s]) / len(refs)) for s in by}      # proportional: hc 115, sc 109, tri 52, acts 24
    rng = random.Random(seed)
    ids = [k for s in sorted(by) for k in rng.sample(sorted(by[s]), quota[s])]
    ids = [k for k in refs if k in set(ids)]                          # keep file order
    json.dump(ids, open(path, "w"))
    return ids


def best_of(res, prefix):
    cands = {k: v for k, v in res.items() if k.startswith(prefix)}
    return max(cands, key=lambda k: cands[k]["bertscore"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "B", "official", "report"])
    ap.add_argument("--config", default=None, help="official: config name from stage B/A")
    ap.add_argument("--official-samples", type=int, default=8, help="official: MBR over seeds 1..N")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if a.stage == "A":
        ids = subset_ids()
        print(f"subset: {len(ids)} prompts", Counter(json.load(open(P('data/corpus/eval/local_ref.json')))[k]['source']
                                                       for k in ids))
        res = evaluate_stage("A", STAGE_A, ids, subset=os.path.join(OUT, "subset300.json"))
        bb, bc = best_of(res, "boost"), best_of(res, "cad")
        combo = {"cad+boost": STAGE_A[bc] + STAGE_A[bb]}
        print(f"best boost {bb}, best CAD {bc} -> {combo}")
        res = evaluate_stage("A", {**STAGE_A, **combo}, ids, subset=os.path.join(OUT, "subset300.json"))
        json.dump({"best_boost": bb, "best_cad": bc, "combo_flags": combo["cad+boost"]},
                  open(os.path.join(OUT, "A", "selection.json"), "w"), indent=1)
    elif a.stage == "B":
        resA = json.load(open(os.path.join(OUT, "A", "metrics.json")))
        top = sorted((k for k in resA if k != "baseline"), key=lambda k: -resA[k]["bertscore"])[:3]
        print("Stage B configs: baseline +", top)
        configs = {"baseline": []} | {k: resA[k]["flags"] for k in top}
        evaluate_stage("B", configs, list(json.load(open(P("data/corpus/eval/local_ref.json")))))
    elif a.stage == "official":
        flags = None
        for st in ("B", "A"):
            p = os.path.join(OUT, st, "metrics.json")
            if os.path.exists(p) and a.config in json.load(open(p)):
                flags = json.load(open(p))[a.config]["flags"]
                break
        assert flags is not None, f"unknown config {a.config}"
        d = os.path.join(OUT, "official")
        mbr = run_config(d, a.config, flags, prompts="official", seeds=list(range(1, a.official_samples + 1)))
        test = json.load(open(P("data", "Phase_1_Pretraining", "Test", "test.json")))
        pred = json.load(open(mbr))
        pred = {k: {"text": pred[k]["text"]} for k in test}           # official id order
        assert len(pred) == len(test) and all(len(v["text"].split()) == 128 for v in pred.values())
        out = os.path.join(d, a.config, "prediction.json")
        json.dump(pred, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.loads(open(out).read())
        import zipfile
        with zipfile.ZipFile(os.path.join(d, a.config, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            z.write(out, arcname="prediction.json")
        print(f"wrote {out}: {len(pred)} ids, all exactly 128 words, valid JSON")
    else:
        import fact_decoding_report
        fact_decoding_report.write(OUT)


if __name__ == "__main__":
    main()
