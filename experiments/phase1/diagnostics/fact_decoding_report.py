"""Tables and plots for eval_fact_decoding.py (called by --stage report).

Writes <out>/TABLES.md, <out>/reuse_vs_distance_A.png (small multiples, <= 4 series per panel) and
<out>/reuse_vs_distance_B.png.
"""

import json
import os

import numpy as np

from attention_reach import BUCKET_NAMES

SOURCES = ["supreme_court", "high_courts", "tribunals", "acts"]
WINDOWS = ["1-32", "33-64", "65-96", "97-128"]
# fixed categorical order (reference palette); baseline always slot 1
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def ci(v, c, p=4):
    return f"{v:.{p}f} ({c[0]:.{p}f}–{c[1]:.{p}f})"


def signed_ci(v, c, p=4):
    return f"{v:+.{p}f} ({c[0]:+.{p}f}…{c[1]:+.{p}f})"


def tables(res, title):
    names = list(res)
    L = [f"## {title}", "", f"n = {res[names[0]]['n']} prompts; 4 samples + MBR per config. "
         "Δ = paired difference to baseline with 95% bootstrap CI. ⚠ = paired CI entirely above 0 (flag).", "",
         "| config | BERTScore F1 (95% CI) | Δ BERTScore vs baseline | ROUGE-L | reuse (all buckets) | invented rate (95% CI) "
         "| closing | header | loops | 8-gram copy (Δ) | short outputs (samples) | s / prompt (4 samples) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        r = res[n]
        L.append(f"| {n} | {ci(r['bertscore'], r['bertscore_ci'])} | "
                 f"{'—' if n == 'baseline' else signed_ci(r['bs_diff'], r['bs_diff_ci'])} | {r['rouge_l']:.4f} | "
                 f"{r['reuse_overall']:.3f} | {ci(r['invented_rate'], r['invented_ci'], 3)} | {r['closing']} | "
                 f"{r['header']} | {r['loops']}{' ⚠' if r['flag_loops'] else ''} | "
                 f"{r['copy8']:.3f}{'' if n == 'baseline' else ' (%+.3f)' % r['copy_diff']}"
                 f"{' ⚠' if r['flag_copy'] else ''} | {r['short']} ({r['short_samples']}) | {r['sec_per_prompt_4samples']:.3f} |")
    L += ["", "BERTScore F1 per 32-word window and per source:", "",
          "| config | " + " | ".join(WINDOWS) + " | " + " | ".join(SOURCES) + " |",
          "|---|" + "---|" * (len(WINDOWS) + len(SOURCES))]
    for n in names:
        r = res[n]
        L.append(f"| {n} | " + " | ".join(f"{r['bertscore_windows'][w]:.4f}" for w in WINDOWS) + " | " +
                 " | ".join(f"{r['bertscore_source'][s]:.4f}" for s in SOURCES) + " |")
    L += ["", "Reuse rate of prompt specifics that the reference also uses, by distance "
          "(tokens from last prompt occurrence to prompt end; 95% cluster-bootstrap CI):", "",
          "| config | " + " | ".join(b for b, c in zip(BUCKET_NAMES, res[names[0]]["reuse"]["n"]) if c) + " |",
          "|---|" + "---|" * sum(1 for c in res[names[0]]["reuse"]["n"] if c)]
    for n in names:
        u = res[n]["reuse"]
        L.append(f"| {n} | " + " | ".join(f"{u['rate'][b]:.3f} ({u['lo'][b]:.3f}–{u['hi'][b]:.3f})"
                                           for b in range(len(BUCKET_NAMES)) if u["n"][b]) + " |")
    L.append("| n specifics | " + " | ".join(str(c) for c in res[names[0]]["reuse"]["n"] if c) + " |")
    L += ["", "Loops / copying deltas vs baseline (paired, 95% CI):", "",
          "| config | Δ loop share | Δ 8-gram copy share | output specifics / output | invented / output |", "|---|---|---|---|---|"]
    for n in names:
        r = res[n]
        L.append(f"| {n} | {signed_ci(r['loop_diff'], r['loop_diff_ci'], 3)} | {signed_ci(r['copy_diff'], r['copy_diff_ci'], 3)} "
                 f"| {r['specifics_per_output']:.2f} | {r['invented_per_output']:.2f} |")
    return L


def panel(ax, res, names, title):
    x = np.arange(len(BUCKET_NAMES))
    for i, n in enumerate(names):
        u = res[n]["reuse"]
        keep = np.array(u["n"]) > 0
        r, lo, hi = (np.array(u[f], dtype=float)[keep] for f in ("rate", "lo", "hi"))
        off = (i - (len(names) - 1) / 2) * 0.1
        ax.errorbar(x[keep] + off, r, yerr=[r - lo, hi - r], color=COLORS[i], lw=2, marker="o", ms=7,
                    capsize=3, mec=SURF, mew=1.5, label=n)
    ax.axvline(3.5, color=INK, lw=1, ls=(0, (1, 3)))
    ax.text(3.45, 0.97, "window = 512", transform=ax.get_xaxis_transform(), va="top", ha="right", fontsize=8, color=MUTED)
    keep = np.array(res[names[0]]["reuse"]["n"]) > 0
    ax.set_xticks(x[keep], [b for b, k in zip(BUCKET_NAMES, keep) if k])
    ax.set_xlim(-0.4, keep.nonzero()[0].max() + 0.4)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("tokens from last prompt occurrence to prompt end", color=MUTED)
    ax.set_ylabel("reuse rate", color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.set_facecolor(SURF)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8, loc="lower left")


def write(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = ["# Fact decoding — result tables", ""]
    A = json.load(open(os.path.join(out, "A", "metrics.json")))
    L += tables(A, "Stage A — 300-prompt stratified subset") + [""]
    groups = [("Prompt boost (b_cont = 4)", ["baseline"] + [k for k in A if k.startswith("boost")]),
              ("CAD alpha (short_len 32, beta 0.1)", ["baseline", "cad_a0.3", "cad_a0.5", "cad_a1.0"]),
              ("CAD short_len (alpha 0.5) and combination", ["baseline", "cad_a0.5_k16", "cad_a0.5_k64", "cad+boost"])]
    groups = [(t, [n for n in g if n in A]) for t, g in groups]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=SURF, sharey=True)
    for ax, (t, g) in zip(axes, groups):
        panel(ax, A, g, t)
    fig.suptitle("Stage A: reuse of prompt specifics by distance (95% CI)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "reuse_vs_distance_A.png"), dpi=140)
    pB = os.path.join(out, "B", "metrics.json")
    if os.path.exists(pB):
        B = json.load(open(pB))
        L += tables(B, "Stage B — all 990 local prompts")
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURF)
        panel(ax, B, list(B)[:4], "Stage B (990 prompts): reuse of prompt specifics by distance (95% CI)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "reuse_vs_distance_B.png"), dpi=140)
    open(os.path.join(out, "TABLES.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
