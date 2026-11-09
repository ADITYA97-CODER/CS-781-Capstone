"""Rerun the Phase 1 attention diagnostics on any checkpoint, and compare checkpoints.

  # generate 4 samples with the submitted settings, MBR-select, score, run the diagnostics
  CUDA_VISIBLE_DEVICES=0 python phase1/diagnostics/eval_checkpoint.py --ckpt runs/archE_diff_145M/ckpt_best.pt \
      --label "Arch E" --out outputs/phase1/diag_archE
  # reuse existing predictions instead of generating (e.g. the submitted Arch D MBR outputs)
  CUDA_VISIBLE_DEVICES=0 python phase1/diagnostics/eval_checkpoint.py --pred outputs/phase1/mbr/mbr_archD4.json \
      --label "Arch D" --out outputs/phase1/diag_archD
  # side-by-side tables + overlay plot
  python phase1/diagnostics/eval_checkpoint.py --compare outputs/phase1/diag_archD outputs/phase1/diag_archE \
      --out outputs/phase1/diag_compare

Submitted setup = generate.py with T=0.7, top-p 0.9, repetition penalty 1.1, batch 128, 320 new tokens,
seeds 1..N, then phase1/mbr/mbr_select.py. Output dir: samples/, mbr.json, scores.txt, diagnostics/.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
P = lambda *a: os.path.join(PROJECT_DIR, *a)
GEN_ARGS = ["--prompts", "local", "--temperature", "0.7", "--top-p", "0.9", "--repetition-penalty", "1.1",
            "--batch-size", "128", "--max-new-tokens", "320"]


def run(cmd, log=None):
    print("$", " ".join(cmd), flush=True)
    with open(log, "w") if log else open(os.devnull, "w") as fh:
        subprocess.run(cmd, check=True, cwd=PROJECT_DIR, stdout=fh if log else None, stderr=subprocess.STDOUT if log else None)


def evaluate(args):
    os.makedirs(args.out, exist_ok=True)
    py = sys.executable
    pred = args.pred
    if pred is None:
        cands = []
        for s in range(1, args.samples + 1):
            d = os.path.join(args.out, "samples", f"s{s}")
            if not os.path.exists(os.path.join(d, "prediction.json")):
                run([py, P("phase1/generate.py"), "--ckpt", args.ckpt, *GEN_ARGS, "--seed", str(s), "--out", d],
                    log=d + ".log")
            cands.append(os.path.join(d, "prediction.json"))
        pred = os.path.join(args.out, "mbr.json")
        if len(cands) > 1:
            run([py, P("phase1/mbr/mbr_select.py"), "--cands", *cands, "--out", pred])
        else:
            json.dump(json.load(open(cands[0])), open(pred, "w"), ensure_ascii=False, indent=1)
    scores = subprocess.run([py, P("official_eval", "Evaluation Script", "evaluate_phase1.py"), "--truth",
                             P("data/corpus/eval/local_ref.json"), "--prediction", pred],
                            capture_output=True, text=True, cwd=PROJECT_DIR).stdout
    scores = "\n".join(l for l in scores.splitlines() if l.startswith(("BERTScore", "ROUGE", "BLEU")))
    open(os.path.join(args.out, "scores.txt"), "w").write(scores + "\n")
    print(scores)
    run([py, P("phase1/diagnostics/attention_reach.py"), "--pred", pred, "--out", os.path.join(args.out, "diagnostics"),
         "--label", args.label])
    json.dump({"ckpt": args.ckpt, "pred": pred, "label": args.label, "samples": args.samples},
              open(os.path.join(args.out, "eval_info.json"), "w"), indent=1)


def load(d):
    r = json.load(open(os.path.join(d, "diagnostics", "results.json")))
    scores = open(os.path.join(d, "scores.txt")).read().strip() if os.path.exists(os.path.join(d, "scores.txt")) else ""
    return r, scores


def compare(args):
    sys.path.insert(0, os.path.dirname(__file__))
    from attention_reach import BUCKET_NAMES
    runs = [load(d) for d in args.compare]
    labels = [r["label"] for r, _ in runs]
    lines = ["# Diagnostics comparison", ""]
    lines += ["## Official metrics (local set)", ""] + [f"- **{l}**: " + s.replace("\n", " · ") for l, (_, s) in zip(labels, runs)]
    lines += ["", "## Reuse rate by distance (95% CI, n)", "", "| bucket | " + " | ".join(labels) + " |",
              "|---|" + "---|" * len(labels)]
    for b, name in enumerate(BUCKET_NAMES):
        cells = []
        for r, _ in runs:
            c = r["curves"]["model"]["all"]
            cells.append("—" if c["n"][b] == 0 else f"{c['rate'][b]:.3f} ({c['lo'][b]:.3f}–{c['hi'][b]:.3f}, n={c['n'][b]})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## Decision rule (512–768 vs trend on 0–512)", ""]
    for l, (r, _) in zip(labels, runs):
        d = r["decision"]["model"]
        lines.append(f"- **{l}**: observed {d['observed_512_768']:.3f}, linear extrapolation {d['extrapolated_512_768']:.3f}, "
                     f"exponential {d.get('extrapolated_512_768_exp', float('nan')):.3f} -> {d['conclusion']}")
    lines += ["", "## Invented specifics (in neither prompt nor reference)", "", "| group | " + " | ".join(labels) + " |",
              "|---|" + "---|" * len(labels)]
    for grp in runs[0][0]["invented"]["model"]:
        cells = []
        for r, _ in runs:
            v = r["invented"]["model"].get(grp, {})
            cells.append("—" if not v.get("n_prompts") else
                         f"{v['invented_rate']:.3f} ({v['ci'][0]:.3f}–{v['ci'][1]:.3f})")
        lines.append(f"| {grp} | " + " | ".join(cells) + " |")
    lines += ["", "## BERTScore F1 per 32-word window (all prompts)", "", "| window | " + " | ".join(labels) + " |",
              "|---|" + "---|" * len(labels)]
    for w in ["1-32", "33-64", "65-96", "97-128"]:
        cells = [f"{r['bertscore_windows']['model']['all'][w]['mean']:.4f}" for r, _ in runs]
        lines.append(f"| {w} | " + " | ".join(cells) + " |")
    os.makedirs(args.out, exist_ok=True)
    open(os.path.join(args.out, "COMPARE.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#2a78d6", "#eb6834", "#1baf7a"]                      # fixed categorical order (max 3 series)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    x = np.arange(len(BUCKET_NAMES))
    for i, ((r, _), l) in enumerate(zip(runs[:3], labels)):
        c = r["curves"]["model"]["all"]
        keep = np.array(c["n"]) > 0
        rate, lo, hi = (np.array(c[f], dtype=float)[keep] for f in ("rate", "lo", "hi"))
        off = (i - (len(runs) - 1) / 2) * 0.12
        ax.errorbar(x[keep] + off, rate, yerr=[rate - lo, hi - rate], color=cols[i], lw=2, marker="o", ms=7,
                    capsize=3, mec="white", mew=1.5, label=l)
    ax.axvline(3.5, color="#0b0b0b", lw=1, ls=(0, (1, 3)))
    ax.text(3.55, 0.97, "window = 512", transform=ax.get_xaxis_transform(), va="top", fontsize=9)
    ax.set_xticks(x, BUCKET_NAMES)
    ax.set_xlabel("tokens from last prompt occurrence to prompt end", color="#52514e")
    ax.set_ylabel("reuse rate (specific appears in output)", color="#52514e")
    ax.set_title("Reuse of prompt specifics by distance (95% bootstrap CI)", loc="left", fontsize=11)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "reuse_vs_distance_compare.png"), dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--pred", default=None, help="existing prediction.json (skips generation)")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", nargs="+", default=None, help="eval_checkpoint output dirs to compare")
    a = ap.parse_args()
    if a.compare:
        compare(a)
    else:
        assert a.ckpt or a.pred, "--ckpt or --pred"
        a.label = a.label or os.path.basename(os.path.dirname(a.ckpt or a.pred))
        evaluate(a)
