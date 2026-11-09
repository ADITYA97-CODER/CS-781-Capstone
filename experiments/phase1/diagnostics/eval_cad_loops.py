"""Reduce CAD's loops while keeping its gains (Arch D doc-mask, 990 local prompts, MBR pipeline).

  CUDA_VISIBLE_DEVICES=1 python phase1/diagnostics/eval_cad_loops.py --stage run       # configs 1-6
  CUDA_VISIBLE_DEVICES=1 python phase1/diagnostics/eval_cad_loops.py --stage temp      # config 7 (best of 3-6 + temp offset)
  CUDA_VISIBLE_DEVICES=1 python phase1/diagnostics/eval_cad_loops.py --stage official --config <name>
  python phase1/diagnostics/eval_cad_loops.py --stage report

Samples: seeds 1-4, T 0.7, top-p 0.9, repetition penalty 1.1, batch 128 (prompt truncation of the submitted
pipeline). Baseline and CAD alpha 0.5 reuse the Stage-B samples (outputs/phase1/fact_decoding/B).
MBR: phase1/mbr/mbr_select.py, with --loop-filter where the config says so.
Metrics: eval_fact_decoding.compute_metrics, paired bootstrap vs baseline.
Recommendation rule (fixed before looking at results): highest BERTScore among configs with
loops <= 1.5 x baseline loops and 8-gram copy rate inside the 95% CI of the references' copy rate.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_fact_decoding import CKPT, Ctx, compute_metrics, ngrams, paired_diff_ci, self_rep  # noqa: E402

PROJECT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
P = lambda *a: os.path.join(PROJECT_DIR, *a)
OUT = P("outputs", "phase1", "cad_loops")
B = P("outputs", "phase1", "fact_decoding", "B")
SEEDS = [1, 2, 3, 4]
GEN = ["--temperature", "0.7", "--top-p", "0.9", "--repetition-penalty", "1.1", "--batch-size", "128",
       "--trunc-group", "128", "--max-new-tokens", "320", "--max-mem-gb", "24"]
CAD = lambda a=0.5: ["--cad", "--cad-alpha", str(a), "--cad-short-len", "32", "--cad-beta", "0.1"]

# sample sets: name -> (generation flags or None for reused samples, list of sample dirs)
SAMPLES = {
    "baseline": (None, [os.path.join(B, "baseline", f"s{s}") for s in SEEDS]),
    "cad": (None, [os.path.join(B, "cad_a0.5", f"s{s}") for s in SEEDS]),
    "baseline_s5-8": (None, [P("outputs", "phase1", "mbr_docmask", f"local_s{s}") for s in (5, 6, 7, 8)]),
    "cad_nrg4": (CAD() + ["--no-repeat-gen-ngram", "4"], None),
    "cad_nrg5": (CAD() + ["--no-repeat-gen-ngram", "5"], None),
    "cad_nrg6": (CAD() + ["--no-repeat-gen-ngram", "6"], None),
    "adapt_a0.5": (CAD(0.5) + ["--cad-adaptive"], None),
    "adapt_a1.0": (CAD(1.0) + ["--cad-adaptive"], None),
}
# MBR configs: name -> (sample sets, loop filter)
CONFIGS = {
    "1 baseline": (["baseline"], False),
    "2 CAD a0.5": (["cad"], False),
    "3 CAD a0.5 + loop filter": (["cad"], True),
    "4 CAD a0.5 + no_repeat_gen 4": (["cad_nrg4"], False),
    "4 CAD a0.5 + no_repeat_gen 5": (["cad_nrg5"], False),
    "4 CAD a0.5 + no_repeat_gen 6": (["cad_nrg6"], False),
    "5 mix 4 base + 4 CAD + loop filter": (["baseline", "cad"], True),
    "6 adaptive a0.5 + loop filter": (["adapt_a0.5"], True),
    "6 adaptive a1.0 + loop filter": (["adapt_a1.0"], True),
    "ctx baseline + loop filter": (["baseline"], True),
    "ctx baseline x8 (s1-8)": (["baseline", "baseline_s5-8"], False),
    "ctx baseline x8 + loop filter": (["baseline", "baseline_s5-8"], True),
}
CANDIDATES_3_6 = [c for c in CONFIGS if c[0] in "3456"]


def sh(cmd, log):
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as fh:
        subprocess.run(cmd, check=True, cwd=PROJECT_DIR, stdout=fh, stderr=subprocess.STDOUT)


def sample_dirs(name, prompts="local"):
    flags, dirs = SAMPLES[name]
    if dirs is not None:
        return dirs
    dirs = []
    for s in SEEDS:
        o = os.path.join(OUT, "samples" if prompts == "local" else "official_samples", name, f"s{s}")
        if not os.path.exists(os.path.join(o, "prediction.json")):
            sh([sys.executable, P("phase1", "generate.py"), "--ckpt", CKPT, "--prompts", prompts, *GEN, *flags,
                "--seed", str(s), "--out", o], o + ".log")
            print(f"  generated {prompts} {name} seed {s}", flush=True)
        dirs.append(o)
    return dirs


def mbr(config, sets, loop_filter, prompts="local"):
    dirs = [d for s in sets for d in sample_dirs(s, prompts)]
    out = os.path.join(OUT, "mbr" if prompts == "local" else "official", config.replace(" ", "_") + ".json")
    if not os.path.exists(out):
        sh([sys.executable, P("phase1", "mbr", "mbr_select.py"), "--cands", *[os.path.join(d, "prediction.json") for d in dirs],
            "--out", out, "--batch-size", "64"] + (["--loop-filter"] if loop_filter else []), out + ".log")
    return out, dirs


def reference_stats(ctx, ids):
    cp = []
    for k in ids:
        g = ngrams(ctx.refs[k]["text"].split(), 8)
        cp.append(sum(x in ctx.p8[k] for x in g) / len(g))
    cp = np.array(cp)
    bi = np.random.default_rng(0).integers(0, len(cp), size=(2000, len(cp)))
    return dict(copy8=float(cp.mean()), copy8_ci=np.percentile(cp[bi].mean(1), [2.5, 97.5]).tolist(),
                loops=int(sum(self_rep(ctx.refs[k]["text"].split()) > 0.25 for k in ids)))


def evaluate(configs):
    os.makedirs(OUT, exist_ok=True)
    paths = {}
    for c, (sets, lf) in configs.items():
        print(f"[{c}]", flush=True)
        paths[c] = mbr(c, sets, lf)
    ctx = Ctx()
    ids = list(ctx.refs)
    res_path = os.path.join(OUT, "metrics.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    res.setdefault("_reference", reference_stats(ctx, ids))
    for c, (p, dirs) in paths.items():
        if c not in res:
            res[c] = compute_metrics(ctx, p, ids, dirs)
            res[c]["n_samples"] = len(dirs)
            json.dump(res, open(res_path, "w"))
            print(f"  {c}: BERTScore {res[c]['bertscore']:.4f} loops {res[c]['loops']} copy8 {res[c]['copy8']:.4f}", flush=True)
    base, ref = res["1 baseline"], res["_reference"]
    for c, r in res.items():
        if c.startswith("_"):
            continue
        for key, field in [("bs", "bertscore_per_prompt"), ("loop", "loops_per_prompt"), ("copy", "copy8_per_prompt")]:
            r[f"{key}_diff"], r[f"{key}_diff_ci"] = paired_diff_ci(r[field], base[field])
        r["eligible"] = bool(r["loops"] <= 1.5 * base["loops"] and ref["copy8_ci"][0] <= r["copy8"] <= ref["copy8_ci"][1])
    json.dump(res, open(res_path, "w"))
    return res


def best_3_6(res):
    pool = [c for c in res if c in CANDIDATES_3_6 or c.startswith("7 ")]
    elig = [c for c in pool if res[c]["eligible"]]
    return max(elig or pool, key=lambda c: res[c]["bertscore"]), bool(elig)


def temp_configs(best):
    """Config 7: the best of 3-6 with the CAD sampling temperature raised by 0.05 / 0.1."""
    sets, lf = CONFIGS[best]
    out = {}
    for off in (0.05, 0.1):
        new_sets = []
        for s in sets:
            if s in ("baseline", "baseline_s5-8"):
                new_sets.append(s)
                continue
            base_flags = SAMPLES[s][0] if SAMPLES[s][0] is not None else CAD()
            name = f"{s}_toff{off}"
            SAMPLES[name] = (base_flags + ["--cad-temp-offset", str(off)], None)
            new_sets.append(name)
        out[f"7 {best[2:]} + temp offset {off}"] = (new_sets, lf)
    return out


def report(res):
    ref = res["_reference"]
    base = res["1 baseline"]
    rows = [c for c in res if not c.startswith("_")]
    L = ["# Reducing CAD's loops (Arch D doc-mask, 990 local prompts)", "",
         f"References: 8-gram copy {ref['copy8']:.4f} (95% CI {ref['copy8_ci'][0]:.4f}–{ref['copy8_ci'][1]:.4f}), loops {ref['loops']}. "
         f"Eligibility: loops ≤ {1.5 * base['loops']:.1f} (1.5 × baseline {base['loops']}) and copy inside the reference CI.", "",
         "| config | samples | BERTScore F1 | Δ vs baseline (95% CI) | ROUGE-L | reuse | invented | loops | 8-gram copy | s/prompt | eligible |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in rows:
        r = res[c]
        d = "—" if c == "1 baseline" else f"{r['bs_diff']:+.4f} ({r['bs_diff_ci'][0]:+.4f}…{r['bs_diff_ci'][1]:+.4f})"
        L.append(f"| {c} | {r['n_samples']} | {r['bertscore']:.4f} | {d} | {r['rouge_l']:.4f} | {r['reuse_overall']:.3f} | "
                 f"{r['invented_rate']:.3f} | {r['loops']} | {r['copy8']:.4f} | {r['sec_per_prompt_4samples']:.3f} | "
                 f"{'yes' if r['eligible'] else 'no'} |")
    L += ["", "| config | 1-32 | 33-64 | 65-96 | 97-128 | supreme_court | high_courts | tribunals | acts |", "|---|---|---|---|---|---|---|---|---|"]
    for c in rows:
        r = res[c]
        L.append(f"| {c} | " + " | ".join(f"{v:.4f}" for v in r["bertscore_windows"].values()) + " | " +
                 " | ".join(f"{r['bertscore_source'][s]:.4f}" for s in ["supreme_court", "high_courts", "tribunals", "acts"]) + " |")
    L += ["", "Reuse by distance (0-128 / 128-256 / 256-384 / 384-512 / 512-768):", ""]
    for c in rows:
        L.append(f"- {c}: " + " / ".join(f"{x:.3f}" for x, n in zip(res[c]["reuse"]["rate"], res[c]["reuse"]["n"]) if n))
    elig = [c for c in rows if res[c]["eligible"] and c != "1 baseline"]
    rec = max(elig, key=lambda c: res[c]["bertscore"]) if elig else None
    L += ["", f"**Recommended (rule above): {rec}**" if rec else "**No config other than baseline meets the rule.**"]
    open(os.path.join(OUT, "RESULTS.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "temp", "report", "official"])
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    if a.stage == "run":
        report(evaluate(CONFIGS))
    elif a.stage == "temp":
        res = json.load(open(os.path.join(OUT, "metrics.json")))
        best, ok = best_3_6({c: r for c, r in res.items() if not c.startswith("_")})
        print(f"best of 3-6: {best} (eligible: {ok})")
        tc = temp_configs(best)
        json.dump({"best_3_6": best, "configs": tc}, open(os.path.join(OUT, "temp_selection.json"), "w"), indent=1)
        report(evaluate({**CONFIGS, **tc}))
    elif a.stage == "report":
        report(json.load(open(os.path.join(OUT, "metrics.json"))))
    else:
        sel = json.load(open(os.path.join(OUT, "temp_selection.json"))) if os.path.exists(os.path.join(OUT, "temp_selection.json")) else {}
        configs = {**CONFIGS, **{k: tuple(v) for k, v in sel.get("configs", {}).items()}}
        if a.config not in configs:
            sys.exit(f"unknown config {a.config!r}")
        if sel:
            temp_configs(sel["best_3_6"])                      # registers the temp-offset sample sets
        sets, lf = configs[a.config]
        # official samples for reused sets: baseline = submitted pipeline samples, cad = Stage-official CAD samples
        SAMPLES["baseline"] = (None, [P("outputs", "phase1", "mbr_docmask", f"official_s{s}") for s in SEEDS])
        SAMPLES["cad"] = (None, [P("outputs", "phase1", "fact_decoding", "official", "cad_a0.5", f"s{s}") for s in SEEDS])
        SAMPLES["baseline_s5-8"] = (None, [P("outputs", "phase1", "mbr_docmask", f"official_s{s}") for s in (5, 6, 7, 8)])
        out, _ = mbr(a.config, sets, lf, prompts="official")
        test = json.load(open(P("data", "Phase_1_Pretraining", "Test", "test.json")))
        pred = json.load(open(out))
        pred = {k: {"text": pred[k]["text"]} for k in test}
        assert len(pred) == 1000 and all(len(v["text"].split()) == 128 for v in pred.values())
        d = os.path.join(OUT, "official", a.config.replace(" ", "_"))
        os.makedirs(d, exist_ok=True)
        json.dump(pred, open(os.path.join(d, "prediction.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        import zipfile
        with zipfile.ZipFile(os.path.join(d, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            z.write(os.path.join(d, "prediction.json"), arcname="prediction.json")
        print(f"wrote {d}/prediction.json: 1000 ids, all 128 words")


if __name__ == "__main__":
    main()
