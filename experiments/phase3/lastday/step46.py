"""Steps 4 + 6: OOF comparison of the final candidates and the candidate submission files (validated).

Candidates (OOF = 2-fold CV over the 90 docs, fixed final epochs, no selection on dev / leaderboard):
  (a) wide 3-config Arch E ensemble (= outputs/phase3/all_wide3 on test)
  (b) (a) + TTA                         -> not run (no saved fold / all-data checkpoints; inference-only impossible)
  (c) (a) + kept rule (Dissent, region v1, relabel variant 'all'; 'ratio_ruling' reported too)
  (d) 0.5750 config + kept rule
  (e) (a) + Arch D LoRMA (equal-weight 4th member) + kept rule
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import rules as R  # noqa: E402

ARCHD = os.path.join(C.LD, "arch_d_lorma")
REF_0575 = os.path.join(C.ROOT, "runs_p3", "e_full_laya_nomarker", "submission.zip")
REF_W3 = os.path.join(C.ROOT, "outputs", "phase3", "all_wide3", "submission.zip")


def with_rule(docs, probs, pred, variant="all", region=R.dissent_region):
    return {k: R.rule_dissent(docs[k]["text"], probs[k], pred[k], variant, region) for k in pred}


def to_labels(pred):
    return {k: [C.LABELS[i] for i in v] for k, v in pred.items()}


def main():
    gold, te = C.gold_docs(), C.test_docs()
    oof = {"wide3": C.mean_probs([C.oof_probs(*C.CONFIGS[n][:2]) for n in C.WIDE3]),
           "full1024": C.oof_probs("full_mlp_la", 10)}
    have_d = all(os.path.exists(os.path.join(ARCHD, f"cv_f{f}", "dev_logp_ep20.npz")) for f in (0, 1))
    if have_d:
        oof["archD"] = C.oof_probs("cv", 20, root=ARCHD)
        oof["wide3+D"] = C.mean_probs([C.oof_probs(*C.CONFIGS[n][:2]) for n in C.WIDE3] + [oof["archD"]])
    cands = {
        "(a) wide3": oof["wide3"], "(c) wide3 + Dissent rule [all]": ("wide3", "all"),
        "(c') wide3 + Dissent rule [ratio_ruling]": ("wide3", "ratio_ruling"),
        "(d0) 0.5750 config, no rule": oof["full1024"], "(d) 0.5750 config + Dissent rule [all]": ("full1024", "all"),
    }
    if have_d:
        cands.update({"Arch D LoRMA alone": oof["archD"], "(e0) wide3 + Arch D, no rule": oof["wide3+D"],
                      "(e) wide3 + Arch D + Dissent rule [all]": ("wide3+D", "all")})
    rows = {}
    for name, c in cands.items():
        if isinstance(c, tuple):
            p = oof[c[0]]
            pred = with_rule(gold, p, C.argmax(p), c[1])
        else:
            pred = C.argmax(c)
        rows[name] = C.score(gold, pred)
    short = {l: l[:9] for l in C.LABELS}
    print(f"{'candidate (OOF)':42s} macro  " + " ".join(f"{short[l]:>9s}" for l in C.LABELS))
    for name, (f1, per) in rows.items():
        print(f"{name:42s} {f1:.4f} " + " ".join(f"{per[l]:9.3f}" for l in C.LABELS))
    json.dump({n: {"macro_f1": f, "per_label": p} for n, (f, p) in rows.items()},
              open(os.path.join(C.LD, "step4_oof.json"), "w"), indent=1)

    # ---------------- Step 6: test files
    tp = {"wide3": C.test_probs([os.path.join(C.ALL, r) for n in C.WIDE3 for r in C.CONFIGS[n][2]])}
    tp_w3_members = [C.test_probs([os.path.join(C.ALL, C.CONFIGS[n][2][0])]) for n in C.WIDE3]
    tp["wide3"] = C.mean_probs(tp_w3_members)
    tp["full1024"] = C.test_probs([os.path.join(C.ROOT, "runs_p3", "e_full_laya_nomarker")])
    ref0575, refw3 = C.load_pred_json(REF_0575), C.load_pred_json(REF_W3)
    assert to_labels(C.argmax(tp["wide3"])) == refw3, "wide3 test probs do not reproduce all_wide3"
    assert to_labels(C.argmax(tp["full1024"])) == ref0575, "test probs do not reproduce the 0.5750 file"
    if os.path.exists(os.path.join(ARCHD, "all", "test_logp.npz")):
        tp["archD"] = C.test_probs([os.path.join(ARCHD, "all")])
        tp["wide3+D"] = C.mean_probs(tp_w3_members + [tp["archD"]])

    def build(out, base, variant="all", region=R.dissent_region):
        p = tp[base]
        before = C.argmax(p)
        after = before if variant is None else with_rule(te, p, before, variant, region)
        lab = to_labels(after)
        zp = C.write_and_validate(lab, os.path.join(C.LD, out))
        changed = [(k, i, C.LABELS[before[k][i]], C.LABELS[after[k][i]]) for k in after
                   for i in np.nonzero(after[k] != before[k])[0]]
        info = {"base": base, "rule": None if variant is None else f"{region.__name__}/{variant}",
                "rule_changed_test_sents": len(changed),
                "changed_vs_0575": C.n_changed(lab, ref0575), "changed_vs_all_wide3": C.n_changed(lab, refw3),
                "label_counts": {l: sum(x.count(l) for x in lab.values()) for l in C.LABELS}}
        json.dump(info, open(os.path.join(C.LD, out, "info.json"), "w"), indent=1)
        print(f"\n{zp}: VALID | rule changed {len(changed)} test sents | vs 0.5750 file {info['changed_vs_0575']} "
              f"| vs all_wide3 {info['changed_vs_all_wide3']} | Dissent {info['label_counts']['Dissent']} PO "
              f"{info['label_counts']['PrecedentOverruled']}")
        for k, i, b, a in changed[:400]:
            print(f"   {k[:40]} #{i}: {b} -> {a} | {te[k]['text'][i][:110]}")
        return info

    oof_rank = {"wide3": rows["(c) wide3 + Dissent rule [all]"][0]}
    if "wide3+D" in tp and have_d:
        oof_rank["wide3+D"] = rows["(e) wide3 + Arch D + Dissent rule [all]"][0]
    best = max(oof_rank, key=oof_rank.get)
    print(f"\nStep 6: cand_cvbest base = {best} (OOF with rule {oof_rank[best]:.4f}; all: {oof_rank})")
    build("cand_cvbest", best)
    build("cand_0575_rules", "full1024")
    # flagged extra files: Dissent region v2 (first-person disagreement cue, written after reading the test text)
    build("cand_cvbest_dissentv2", best, "all", R.dissent_region_v2)
    build("cand_0575_dissentv2", "full1024", "all", R.dissent_region_v2)
    if "archD" in tp:
        build("arch_d_lorma", "archD", None)


if __name__ == "__main__":
    main()
