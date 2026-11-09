"""Section 5: evaluate the masked-task-SFT runs against their starting checkpoint on the 990 local prompts.

Decoding = the current best pipeline WITHOUT CAD, identical for every model:
  generate.py --temperature 0.7 --top-p 0.9 --repetition-penalty 1.1 --batch-size 128 --max-new-tokens 320
              --max-mem-gb 26 --steer-headers --steer-cite, seeds 1..8
  + phase1/mbr/mbr_rerank.py with the fixed penalty weights of outputs/phase1/mbr_rerank/archE_local_rerank8_tuning.json

Two steps: first ONE sample per prompt (seed 1, same sampling + steering, no MBR) for all
models; only if the SFT models look promising vs the start model, the full 8-sample MBR (seed 1 is reused).

  python phase1/sft/eval_masked_sft.py --stage single --gpus 0,1     # seed 1 for all models + REPORT_single.md
  python phase1/sft/eval_masked_sft.py --stage gen --gpus 0,1        # seeds 1-8 + reranked MBR outputs
  python phase1/sft/eval_masked_sft.py --stage report                # metrics, CIs, plots, REPORT.md (MBR x8)
  python phase1/sft/eval_masked_sft.py --stage official --model sft_unsupported --gpus 0,1   # NEW prediction.json

Outputs: outputs/phase1/masked_sft/eval/<model>/s<seed>/, <model>_mbr8.json, metrics.json, REPORT.md, plots/*.png
"""
import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time

import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_DIR, "phase1", "diagnostics"))
P = lambda *a: os.path.join(PROJECT_DIR, *a)
OUT = P("outputs", "phase1", "masked_sft", "eval")
PY = sys.executable
SEEDS = list(range(1, 9))
MODELS = {"start": "runs/archE_diff_145M/ckpt_best.pt",
          "sft_none": "runs/sft_archE_v2_none/ckpt_best.pt",
          "sft_unsupported": "runs/sft_archE_v2_unsupported/ckpt_best.pt",
          "sft_upweight": "runs/sft_archE_v2_upweight/ckpt_best.pt"}
LABEL = {"start": "start (Arch E)", "sft_none": "SFT none", "sft_unsupported": "SFT unsupported",
         "sft_upweight": "SFT unsupported+upweight"}
COLOR = {"start": "#2a78d6", "sft_none": "#eb6834", "sft_unsupported": "#1baf7a", "sft_upweight": "#eda100"}
GEN = ["--temperature", "0.7", "--top-p", "0.9", "--repetition-penalty", "1.1", "--batch-size", "128",
       "--max-new-tokens", "320", "--max-mem-gb", "26", "--steer-headers", "--steer-cite"]
WEIGHTS = P("outputs", "phase1", "mbr_rerank", "archE_local_rerank8_tuning.json")
JUNK = re.compile(r"(?:/[a-z]{1,6}){3,}|(?:\b[a-z]{1,3}/){3,}")


def run_jobs(jobs, gpus):
    """jobs: list of (cmd, log, done_file). One worker thread per GPU pulls jobs from a queue."""
    q = queue.Queue()
    for j in jobs:
        if not os.path.exists(j[2]):
            q.put(j)
    print(f"{q.qsize()} generation jobs on GPUs {gpus}", flush=True)
    errors = []

    def worker(g):
        while True:
            try:
                cmd, log, _ = q.get_nowait()
            except queue.Empty:
                return
            t0 = time.time()
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
            with open(log, "w") as fh:
                r = subprocess.run(cmd, cwd=PROJECT_DIR, env=env, stdout=fh, stderr=subprocess.STDOUT)
            print(f"{time.strftime('%T')} GPU {g} {'ok' if r.returncode == 0 else 'FAILED'} "
                  f"{os.path.basename(os.path.dirname(log))}/{os.path.basename(log)} {time.time() - t0:.0f}s", flush=True)
            if r.returncode:
                errors.append(log)
    th = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if errors:
        sys.exit(f"failed: {errors}")


def gen_jobs(model, ckpt, prompts, outdir, seeds=SEEDS):
    jobs = []
    for s in seeds:
        d = os.path.join(outdir, model, f"s{s}")
        os.makedirs(os.path.dirname(d), exist_ok=True)
        jobs.append(([PY, P("phase1/generate.py"), "--ckpt", P(ckpt), "--prompts", prompts, *GEN, "--seed", str(s),
                      "--out", d], d + ".log", os.path.join(d, "prediction.json")))
    return jobs


def rerank(model, prompts, outdir):
    out = os.path.join(outdir, f"{model}_mbr8.json")
    if not os.path.exists(out):
        w = json.load(open(WEIGHTS))["weights"]
        subprocess.run([PY, P("phase1/mbr/mbr_rerank.py"), "--cands",
                        *[os.path.join(outdir, model, f"s{s}", "prediction.json") for s in SEEDS],
                        "--prompts", prompts, "--weights", json.dumps(w), "--out", out, "--batch-size", "64"],
                       check=True, cwd=PROJECT_DIR, env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("RERANK_GPU", "0")},
                       stdout=subprocess.DEVNULL)
    return out


def stage_gen(a, single=False):
    gpus = [int(g) for g in a.gpus.split(",")]
    models = a.models.split(",") if a.models else list(MODELS)
    for m in models:
        if not os.path.exists(P(MODELS[m])):
            sys.exit(f"missing checkpoint for {m}: {MODELS[m]}")
    jobs = [j for m in models for j in gen_jobs(m, MODELS[m], "local", OUT, [1] if single else SEEDS)]
    run_jobs(jobs, gpus)
    if single:
        return
    os.environ["RERANK_GPU"] = str(gpus[0])
    for m in models:
        print("rerank", m, rerank(m, "local", OUT), flush=True)


def stage_single(a):
    """Seed 1 only (sampling + steering, no MBR) for every model, then the same report on those single samples."""
    stage_gen(a, single=True)
    stage_report(a, mode="single")


# ----------------------------------------------------------------------------- report

def stage_report(a, mode="mbr8"):
    from attention_reach import regex_specifics
    from eval_fact_decoding import Ctx, compute_metrics, ngrams, paired_diff_ci, SOURCES
    sys.path.insert(0, P("phase1", "mbr"))
    from mbr_rerank import SIGN
    ctx = Ctx()
    ids = list(ctx.refs)
    single = mode == "single"
    pred_path = (lambda m: os.path.join(OUT, m, "s1", "prediction.json")) if single else (lambda m: os.path.join(OUT, f"{m}_mbr8.json"))
    models = [m for m in MODELS if os.path.exists(pred_path(m))]
    if "start" not in models:
        sys.exit("start-model outputs missing")
    res, txts = {}, {}
    for m in models:
        path = pred_path(m)
        dirs = [os.path.join(OUT, m, f"s{s}") for s in ([1] if single else SEEDS)]
        res[m] = compute_metrics(ctx, path, ids, dirs)
        pred = json.load(open(path))
        txts[m] = [pred[k]["text"] for k in ids]
        print("metrics", m, f"{res[m]['bertscore']:.4f}", flush=True)
    refs = [ctx.refs[k]["text"] for k in ids]
    src = np.array([ctx.refs[k]["source"] for k in ids])

    # per-prompt BERTScore of the four 32-word windows (for paired window CIs)
    def windows(txt):
        cand, ref = [], []
        for a_ in range(0, 128, 32):
            cand += [" ".join(t.split()[a_:a_ + 32]) or "." for t in txt]
            ref += [" ".join(r.split()[a_:a_ + 32]) or "." for r in refs]
        _, _, f = ctx.scorer.score(cand, ref)
        return np.array(f.tolist()).reshape(4, len(ids))
    win = {m: windows(txts[m]) for m in models}

    # extra per-output measures (also on the references, as their natural rate)
    out_ner = {m: ctx.run_ner(txts[m], ids) for m in models}

    def extras(txt, ner, is_ref=False):
        inv_cit = inv_date = no_spec = junk = sign = 0
        n_cit = n_date = 0
        for i, k in enumerate(ids):
            t = txt[i]
            rx = regex_specifics(t)
            sp = ctx.specifics(t, ner[k], rx)
            if not sp:
                no_spec += 1
            p_keys = ctx.regex[("prompt", k)]
            r_keys = ctx.regex[("ref", k)]
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
        return dict(no_specifics_share=no_spec / len(ids), citations=n_cit, invented_citations=inv_cit,
                    dates=n_date, invented_dates=inv_date, junk=junk, signature_blocks=sign)

    from eval_fact_decoding import CLOSERS, HEADERS, self_rep
    ref_ex = extras(refs, ctx.ner["ref"], is_ref=True)
    ref_cp = [np.mean([x in ctx.p8[k] for x in ngrams(refs[i].split(), 8)]) for i, k in enumerate(ids)]
    ref_rates = dict(closing=sum(bool(CLOSERS.search(t)) for t in refs), header=sum(bool(HEADERS.search(t)) for t in refs),
                     loops=sum(self_rep(t.split()) > 0.25 for t in refs), copy8=float(np.mean(ref_cp)), **ref_ex)
    base = res["start"]
    for m in models:
        r = res[m]
        r.update(extras(txts[m], out_ner[m]))
        r["window_per_prompt"] = win[m]
        if m == "start":
            continue
        r["bs_diff"], r["bs_diff_ci"] = paired_diff_ci(r["bertscore_per_prompt"], base["bertscore_per_prompt"])
        r["bs_diff_source"] = {s: paired_diff_ci(np.array(r["bertscore_per_prompt"])[src == s],
                                                 np.array(base["bertscore_per_prompt"])[src == s]) for s in SOURCES}
        r["bs_diff_window"] = {f"{a_ * 32 + 1}-{a_ * 32 + 32}": paired_diff_ci(win[m][a_], win["start"][a_]) for a_ in range(4)}
        r["loop_diff"], r["loop_diff_ci"] = paired_diff_ci(r["loops_per_prompt"], base["loops_per_prompt"])
        r["copy_diff"], r["copy_diff_ci"] = paired_diff_ci(r["copy8_per_prompt"], base["copy8_per_prompt"])
        r["no_spec_diff"] = r["no_specifics_share"] - base["no_specifics_share"]
        # recommendation rule (fixed in advance): CI excludes 0 upward and no sharp rise in outputs without specifics
        r["recommendable"] = bool(r["bs_diff_ci"][0] > 0 and r["no_spec_diff"] <= 0.05)

    suffix = "_single" if single else ""
    os.makedirs(os.path.join(OUT, "plots" + suffix), exist_ok=True)
    plots(res, models, "plots" + suffix)
    L = report_md(res, models, ref_rates, single)
    open(os.path.join(OUT, f"REPORT{suffix}.md"), "w").write("\n".join(L) + "\n")
    for r in res.values():
        for k in ("bertscore_per_prompt", "loops_per_prompt", "copy8_per_prompt", "reuse_hits_tots", "window_per_prompt"):
            r.pop(k, None)
    json.dump(dict(results=res, reference_rates=ref_rates), open(os.path.join(OUT, f"metrics{suffix}.json"), "w"), indent=1, default=float)
    print("\n".join(L))


def plots(res, models, pdir="plots"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8})
    sft = [m for m in models if m != "start"]
    # 1. BERTScore difference vs start per 32-word window, with 95% CI
    fig, ax = plt.subplots(figsize=(7, 3.8))
    wins = ["1-32", "33-64", "65-96", "97-128"]
    for j, m in enumerate(sft):
        d = [res[m]["bs_diff_window"][w] for w in wins]
        x = np.arange(4) + (j - (len(sft) - 1) / 2) * 0.12
        ax.errorbar(x, [v[0] for v in d], yerr=[[v[0] - v[1][0] for v in d], [v[1][1] - v[0] for v in d]],
                    fmt="o", ms=8, lw=2, capsize=3, color=COLOR[m], label=LABEL[m])
    ax.axhline(0, color="#52514e", lw=1)
    ax.set_xticks(range(4), [f"words {w}" for w in wins])
    ax.set_ylabel("BERTScore F1 difference vs start")
    ax.set_title("Difference vs starting checkpoint by word window (95% CI)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, pdir, "bertscore_diff_windows.png"), dpi=150); plt.close(fig)
    # 2. per source
    fig, ax = plt.subplots(figsize=(7, 3.8))
    srcs = ["supreme_court", "high_courts", "tribunals", "acts"]
    for j, m in enumerate(sft):
        d = [res[m]["bs_diff_source"][s] for s in srcs]
        x = np.arange(4) + (j - (len(sft) - 1) / 2) * 0.12
        ax.errorbar(x, [v[0] for v in d], yerr=[[v[0] - v[1][0] for v in d], [v[1][1] - v[0] for v in d]],
                    fmt="o", ms=8, lw=2, capsize=3, color=COLOR[m], label=LABEL[m])
    ax.axhline(0, color="#52514e", lw=1)
    ax.set_xticks(range(4), ["Supreme Court", "High Courts", "Tribunals", "Acts"])
    ax.set_ylabel("BERTScore F1 difference vs start")
    ax.set_title("Difference vs starting checkpoint by source (95% CI)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, pdir, "bertscore_diff_sources.png"), dpi=150); plt.close(fig)
    # 3. reuse by distance
    from attention_reach import BUCKET_NAMES
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for m in models:
        r = res[m]["reuse"]
        keep = [i for i, n in enumerate(r["n"]) if n >= 20]
        ax.plot(keep, [r["rate"][i] for i in keep], "-o", lw=2, ms=8, color=COLOR[m], label=LABEL[m],
                markeredgecolor="white", markeredgewidth=1.5)
    ax.set_xticks(keep, [BUCKET_NAMES[i] for i in keep])
    ax.set_xlabel("distance of the specific's last prompt occurrence from the prompt end (tokens)")
    ax.set_ylabel("share reused in output")
    ax.set_title("Reuse of prompt specifics needed by the reference", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, pdir, "reuse_by_distance.png"), dpi=150); plt.close(fig)


def report_md(res, models, ref, single=False):
    f4 = lambda v: f"{v:+.4f}"
    ci = lambda d, c: f"{f4(d)} ({f4(c[0])}…{f4(c[1])})"
    dec = ("ONE sample per prompt (seed 1; T 0.7, top-p 0.9, rep 1.1, header + citation steering), no MBR" if single else
           "8 samples (T 0.7, top-p 0.9, rep 1.1, header + citation steering) + penalty-aware MBR (fixed weights)")
    L = [f"# Masked task SFT — evaluation on the 990 local prompts ({'single sample' if single else 'MBR x8'})", "",
         f"Decoding for every model: {dec}. No CAD. Δ = paired difference vs the starting checkpoint (Arch E) "
         "with the same decoding, 95% bootstrap CI over prompts.", "",
         "| model | BERTScore | Δ vs start (95% CI) | ROUGE-L | reuse | invented rate | invented / output | no-specifics share | invented citations | invented dates |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        r = res[m]
        d = "—" if m == "start" else ci(r["bs_diff"], r["bs_diff_ci"])
        L.append(f"| {LABEL[m]} | {r['bertscore']:.4f} | {d} | {r['rouge_l']:.4f} | {r['reuse_overall']:.3f} | "
                 f"{r['invented_rate']:.3f} | {r['invented_per_output']:.2f} | {r['no_specifics_share']:.3f} | "
                 f"{r['invented_citations']}/{r['citations']} | {r['invented_dates']}/{r['dates']} |")
    L.append(f"| references | — | — | — | — | — | — | {ref['no_specifics_share']:.3f} | (new vs prompt) {ref['invented_citations']}/{ref['citations']} | "
             f"{ref['invented_dates']}/{ref['dates']} |")
    L += ["", "| model | closing formulas | court headers | signature blocks | samples < 128 words | outputs < 128 words | loops | junk | 8-gram prompt copy |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        r = res[m]
        L.append(f"| {LABEL[m]} | {r['closing']} | {r['header']} | {r['signature_blocks']} | {r['short_samples']} | {r['short']} | "
                 f"{r['loops']} | {r['junk']} | {r['copy8']:.4f} |")
    L.append(f"| references | {ref['closing']} | {ref['header']} | {ref['signature_blocks']} | — | — | {ref['loops']} | {ref['junk']} | {ref['copy8']:.4f} |")
    L += ["", "BERTScore Δ vs start per source and per 32-word window (95% CI):", "",
          "| model | " + " | ".join(["Supreme Court", "High Courts", "Tribunals", "Acts", "w 1-32", "w 33-64", "w 65-96", "w 97-128"]) + " |",
          "|---|" + "---|" * 8]
    for m in models:
        if m == "start":
            continue
        r = res[m]
        cells = [ci(*r["bs_diff_source"][s]) for s in ["supreme_court", "high_courts", "tribunals", "acts"]]
        cells += [ci(*r["bs_diff_window"][w]) for w in ["1-32", "33-64", "65-96", "97-128"]]
        L.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    L += ["", "Absolute BERTScore per source / window:", ""]
    for m in models:
        r = res[m]
        L.append(f"- {LABEL[m]}: " + ", ".join(f"{k} {v:.4f}" for k, v in r["bertscore_source"].items()) + " | "
                 + ", ".join(f"{k} {v:.4f}" for k, v in r["bertscore_windows"].items()))
    from attention_reach import BUCKET_NAMES
    L += ["", "Reuse by distance bucket (tokens from prompt end): " + " / ".join(BUCKET_NAMES), ""]
    for m in models:
        rr = res[m]["reuse"]
        L.append(f"- {LABEL[m]}: " + " / ".join(f"{x:.3f} (n {c})" for x, c in zip(rr["rate"], rr["n"]) if c))
    L += ["", "Loops / copy Δ vs start (95% CI):", ""]
    for m in models:
        if m != "start":
            r = res[m]
            L.append(f"- {LABEL[m]}: loops {ci(r['loop_diff'], r['loop_diff_ci'])}, 8-gram copy {ci(r['copy_diff'], r['copy_diff_ci'])}, "
                     f"no-specifics share {r['no_spec_diff']:+.3f}")
    pd_ = "plots_single" if single else "plots"
    L += ["", f"Plots: `{pd_}/bertscore_diff_windows.png`, `{pd_}/bertscore_diff_sources.png`, `{pd_}/reuse_by_distance.png`", ""]
    if single:
        L += ["## Gate for the 8-sample MBR run (single-sample screen)",
              "Promising = BERTScore Δ vs start > 0 (point estimate; single samples are noisy, so the CI is reported, not required), "
              "invented rate not higher than the start model's, and no-specifics share up by at most 5 points.", ""]
        for m in models:
            if m == "start":
                continue
            r = res[m]
            ok = r["bs_diff"] > 0 and r["invented_rate"] <= res["start"]["invented_rate"] and r["no_spec_diff"] <= 0.05
            L.append(f"- {LABEL[m]}: {'PROMISING' if ok else 'not promising'} (Δ {ci(r['bs_diff'], r['bs_diff_ci'])}, invented "
                     f"{r['invented_rate']:.3f} vs {res['start']['invented_rate']:.3f}, no-specifics {r['no_spec_diff']:+.3f})")
        return L
    L += ["## Recommendation rule (fixed before looking at results)",
          "A run is recommended only if its BERTScore Δ vs start has a 95% CI entirely above 0 AND its share of outputs with no "
          "specifics rises by at most 5 percentage points. Among qualifying runs, the highest BERTScore.", ""]
    ok = [m for m in models if m != "start" and res[m].get("recommendable")]
    if ok:
        best = max(ok, key=lambda m: res[m]["bertscore"])
        L.append(f"**Recommended: {LABEL[best]}** ({res[best]['bertscore']:.4f}, Δ {ci(res[best]['bs_diff'], res[best]['bs_diff_ci'])}).")
    else:
        L.append("**No run qualifies; keep the starting checkpoint.**")
    return L


# ----------------------------------------------------------------------------- official

def stage_official(a):
    m = a.model
    gpus = [int(g) for g in a.gpus.split(",")]
    od = P("outputs", "phase1", "masked_sft", "official", m)
    if os.path.exists(os.path.join(od, "prediction.json")):
        sys.exit(f"{od}/prediction.json exists; not overwriting")
    run_jobs(gen_jobs(m, MODELS[m], "official", od), gpus)
    os.environ["RERANK_GPU"] = str(gpus[0])
    mbr = rerank(m, "official", od)
    test = json.load(open(P("data", "Phase_1_Pretraining", "Test", "test.json")))
    pred = json.load(open(mbr))
    short = [k for k in test if len(pred[k]["text"].split()) < 128]
    if short:
        print("WARNING: outputs under 128 words (regenerate these with --max-new-tokens 480):", short)
    pred = {k: {"text": " ".join(pred[k]["text"].split()[:128])} for k in test}
    json.dump(pred, open(os.path.join(od, "prediction.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("wrote", os.path.join(od, "prediction.json"), len(pred), "ids; short:", short)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["single", "gen", "report", "official"])
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--models", default=None, help="comma list (default all)")
    ap.add_argument("--model", default=None, help="official stage: which model")
    a = ap.parse_args()
    {"single": stage_single, "gen": stage_gen, "report": stage_report, "official": stage_official}[a.stage](a)


if __name__ == "__main__":
    main()
