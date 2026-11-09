"""Summarise an LR sweep (runs/<prefix><lr>/metrics.jsonl): loss / grad-norm curves, rejections, choice.

  python phase1/diagnostics/lr_sweep_report.py --runs runs/archE_sweep_lr0.001 runs/archE_sweep_lr0.002 ... \
      --out outputs/phase1/lr_sweep

Rules (fixed before looking at the results):
  score        mean training loss over the last 50 steps (validation loss at the end is also shown)
  loss spike   after warmup, a step loss more than 0.15 above the median of the previous 20 steps
  rising gn    over the second half of the run, grad norm has a positive trend (slope of a linear fit
               on log grad norm > 0) AND its last-50-step median is >10% above the minimum
               50-step median seen after warmup
  choice       lowest score among LRs without spikes / rising grad norm; if the best two are within
               0.01, the lower LR
"""

import argparse
import json
import os

import numpy as np


def load(run):
    recs = [json.loads(l) for l in open(os.path.join(run, "metrics.jsonl"))]
    cfg = json.load(open(os.path.join(run, "config.json")))
    train = [r for r in recs if "loss" in r]
    val = [r for r in recs if "val" in r]
    return cfg, train, val


def analyse(run):
    cfg, train, val = load(run)
    warm = cfg["train"]["warmup_steps"]
    steps = np.array([r["step"] for r in train])
    loss = np.array([r["loss"] for r in train])
    gn = np.array([r["grad_norm"] for r in train])
    spikes = []
    for i in range(len(loss)):
        if steps[i] <= warm or i < 20:
            continue
        if loss[i] > np.median(loss[i - 20:i]) + 0.15:
            spikes.append(int(steps[i]))
    half = steps >= steps[-1] / 2
    slope = np.polyfit(steps[half], np.log(gn[half]), 1)[0]
    med50 = [np.median(gn[max(0, i - 50):i]) for i in range(50, len(gn) + 1, 10) if steps[i - 1] > warm]
    rising = bool(slope > 0 and med50 and np.median(gn[-50:]) > 1.10 * min(med50))
    return {"run": run, "lr": cfg["train"]["lr"], "steps": int(steps[-1]) + 1,
            "score_last50": float(loss[-50:].mean()), "final_val": val[-1]["val"]["mean"] if val else None,
            "final_val_bm": val[-1].get("val_bm", {}).get("mean") if val else None,
            "spikes": spikes, "gn_slope_2nd_half": float(slope), "gn_last50_median": float(np.median(gn[-50:])),
            "rising_grad_norm": rising, "rejected": bool(spikes) or rising,
            "tok_per_s": train[-1]["tok_per_s"], "mfu": train[-1].get("mfu"),
            "lambda_last": train[-1].get("lambda"),
            "_curves": (steps.tolist(), loss.tolist(), gn.tolist())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = sorted((analyse(r) for r in args.runs), key=lambda r: r["lr"])
    ok = sorted([r for r in res if not r["rejected"]], key=lambda r: r["score_last50"])
    choice = None
    if ok:
        choice = ok[0]
        if len(ok) > 1 and ok[1]["score_last50"] - ok[0]["score_last50"] < 0.01 and ok[1]["lr"] < ok[0]["lr"]:
            choice = ok[1]
    os.makedirs(args.out, exist_ok=True)
    lines = ["| peak LR | steps | train loss (last 50) | val loss | val_bm | spikes | grad-norm slope (2nd half) | "
             "gn median (last 50) | rejected |", "|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        lines.append(f"| {r['lr']:.0e} | {r['steps']} | {r['score_last50']:.4f} | {r['final_val']:.4f} | "
                     f"{(r['final_val_bm'] or float('nan')):.4f} | {r['spikes'] or 'none'} | "
                     f"{r['gn_slope_2nd_half']:+.2e} | {r['gn_last50_median']:.3f} | {'yes' if r['rejected'] else 'no'} |")
    lines.append("")
    lines.append(f"**Chosen peak LR: {choice['lr']:.0e}**" if choice else "**All LRs rejected**")
    open(os.path.join(args.out, "SWEEP.md"), "w").write("\n".join(lines) + "\n")
    json.dump([{k: v for k, v in r.items() if k != "_curves"} for r in res] + [{"choice": choice and choice["lr"]}],
              open(os.path.join(args.out, "sweep.json"), "w"), indent=1)
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), facecolor="#fcfcfb")
    for i, r in enumerate(res[:3]):
        s, l, g = map(np.array, r["_curves"])
        k = 10                                                    # light smoothing for readability
        ls = np.convolve(l, np.ones(k) / k, mode="valid")
        axes[0].plot(s[k - 1:], ls, color=cols[i], lw=2, label=f"lr {r['lr']:.0e}")
        axes[0].plot(s, l, color=cols[i], lw=0.6, alpha=0.3)
        axes[1].plot(s, g, color=cols[i], lw=1.2, label=f"lr {r['lr']:.0e}")
    axes[0].set_ylim(None, min(6, max(np.array(res[0]["_curves"][1])[50:].max(), 3)))
    axes[0].set_title("Training loss (10-step mean; raw faint)", loc="left", fontsize=11)
    axes[1].set_title("Gradient norm (before clipping at 1.0)", loc="left", fontsize=11)
    axes[1].set_ylim(0, min(3, max(max(r["_curves"][2][30:]) for r in res) * 1.1))
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.set_xlabel("step", color="#52514e")
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "lr_sweep.png"), dpi=150)


if __name__ == "__main__":
    main()
