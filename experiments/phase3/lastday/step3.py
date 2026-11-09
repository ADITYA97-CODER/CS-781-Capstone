"""Step 3: evaluate the rare-label rules on OOF predictions (wide3 ensemble and the 0.5750 config), keep/reject by
the user's criterion (OOF macro-F1 up AND no other label's F1 down by > 0.01), and list test sentences they change."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import rules as R  # noqa: E402

RULES = {
    "po_argmax": lambda t, p, y: R.rule_precedent_overruled(t, p, y, "argmax"),
    "po_top2": lambda t, p, y: R.rule_precedent_overruled(t, p, y, "top2"),
    "dissent_ratio_ruling": lambda t, p, y: R.rule_dissent(t, p, y, "ratio_ruling"),
    "dissent_all": lambda t, p, y: R.rule_dissent(t, p, y, "all"),
    "dissent2_ratio_ruling": lambda t, p, y: R.rule_dissent(t, p, y, "ratio_ruling", R.dissent_region_v2),
    "dissent2_all": lambda t, p, y: R.rule_dissent(t, p, y, "all", R.dissent_region_v2),
}
TARGET = {"po": C.PO, "dissent": C.DIS, "dissent2": C.DIS}


def apply(rule, docs, probs, pred):
    return {k: RULES[rule](docs[k]["text"], probs[k], pred[k]) for k in pred}


def evaluate(name, gold, probs):
    base = C.argmax(probs)
    f0, per0 = C.score(gold, base)
    print(f"\n== {name}: base OOF macro-F1 {f0:.4f}")
    res = {}
    for r in RULES:
        new = apply(r, gold, probs, base)
        f1, per = C.score(gold, new)
        tgt = C.LABELS[TARGET[r.split("_")[0]]]
        drops = {l: round(per[l] - per0[l], 4) for l in C.LABELS if l != tgt and per[l] - per0[l] < -1e-9}
        worst = min(drops.values(), default=0.0)
        changed = sum(int((new[k] != base[k]).sum()) for k in base)
        # rule precision/recall as a detector of the target label (sentences it set to the target label)
        y = np.concatenate([gold[k]["y"] for k in sorted(gold)])
        fired = np.concatenate([(new[k] != base[k]) for k in sorted(gold)])
        tp = int((fired & (y == TARGET[r.split("_")[0]])).sum())
        prec = tp / fired.sum() if fired.sum() else 0.0
        rec = tp / (y == TARGET[r.split("_")[0]]).sum()
        n_docs = sum(int((new[k] != base[k]).any()) for k in base)
        keep = f1 > f0 + 1e-9 and worst >= -0.01
        res[r] = {"macro_f1": f1, "delta": f1 - f0, "target_f1_before": per0[tgt], "target_f1_after": per[tgt],
                  "changed": changed, "docs_changed": n_docs, "rule_precision": prec, "rule_recall": rec,
                  "other_label_drops": drops, "keep": keep}
        print(f"  {r:22s} macro {f1:.4f} ({f1 - f0:+.4f}) | {tgt} F1 {per0[tgt]:.3f} -> {per[tgt]:.3f} | changed "
              f"{changed} sents in {n_docs} docs, precision {prec:.3f} recall {rec:.3f} | other drops {drops} | "
              f"{'KEEP' if keep else 'reject'}")
    return res


def main():
    gold, te = C.gold_docs(), C.test_docs()
    # detector-level stats (independent of the model)
    gate = [(k, i, gold[k]["labels"][i]) for k in gold for i, s in enumerate(gold[k]["text"]) if R.po_gate(s)]
    n_po = sum(int((d["y"] == C.PO).sum()) for d in gold.values())
    print(f"PO gate on 90 docs: fires {len(gate)} sents, {sum(g[2] == 'PrecedentOverruled' for g in gate)} gold PO "
          f"(of {n_po}); gold labels of hits {sorted(set(g[2] for g in gate))}")
    print(f"PO gate on test: fires {sum(R.po_gate(s) for d in te.values() for s in d['text'])} sents")
    for name, docs in (("90 labelled", gold), ("test", te)):
        for k, d in docs.items():
            reg = R.dissent_region(d["text"])
            reg2 = R.dissent_region_v2(d["text"])
            if reg2 != reg:
                print(f"Dissent region v2 only [{name}] {k}: sents {reg2[0]}-{reg2[1] - 1} of {len(d['text'])}")
            if reg:
                extra = ""
                if d.get("labels"):
                    g = d["y"][reg[0]:reg[1]]
                    extra = f", gold Dissent in region {int((g == C.DIS).sum())}/{reg[1] - reg[0]}, " \
                            f"gold Dissent in doc {int((d['y'] == C.DIS).sum())}"
                print(f"Dissent region fires [{name}] {k}: sents {reg[0]}-{reg[1] - 1} of {len(d['text'])}{extra}")
                print("   opening:", d["text"][reg[0]][:160])
    out = {}
    wide = C.mean_probs([C.oof_probs(*C.CONFIGS[n][:2]) for n in C.WIDE3])
    out["wide3"] = evaluate("wide3 ensemble (OOF)", gold, wide)
    out["full_mlp_la_1024"] = evaluate("0.5750 config E_full_mlp_la_1024 (OOF)", gold, C.oof_probs("full_mlp_la", 10))
    json.dump(out, open(os.path.join(C.LD, "step3_rules.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
