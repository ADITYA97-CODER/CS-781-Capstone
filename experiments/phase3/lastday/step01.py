"""Step 0 (label statistics over the 90 labelled docs) and Step 1 (tidy OOF + test probability store)."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def step0(gold):
    rows = []
    for l in C.LABELS:
        k = C.D.LAB2ID[l]
        r = {"label": l}
        for dom in ("CL", "IT"):
            ds = [d for d in gold.values() if d["domain"] == dom]
            r[f"sents_{dom}"] = int(sum((d["y"] == k).sum() for d in ds))
            r[f"docs_{dom}"] = int(sum((d["y"] == k).any() for d in ds))
        r["sents"] = r["sents_CL"] + r["sents_IT"]
        r["docs"] = r["docs_CL"] + r["docs_IT"]
        rows.append(r)
    t = pd.DataFrame(rows)[["label", "sents", "docs", "sents_CL", "docs_CL", "sents_IT", "docs_IT"]]
    n = {dom: sum(d["domain"] == dom for d in gold.values()) for dom in ("CL", "IT")}
    print(f"90 labelled docs: CL {n['CL']}, IT {n['IT']}")
    print(t.to_string(index=False))
    t.to_csv(os.path.join(C.LD, "step0_label_stats.csv"), index=False)
    for l in ("Dissent", "PrecedentOverruled"):
        k = C.D.LAB2ID[l]
        print(l, "docs:", [(d["id"], d["domain"], int((d["y"] == k).sum())) for d in gold.values() if (d["y"] == k).any()])


def tidy(probs, gold_or_test, config, arch, split):
    recs = []
    for doc_id, p in probs.items():
        d = gold_or_test[doc_id]
        for i in range(len(p)):
            recs.append((doc_id, i, d["domain"], d["labels"][i] if d.get("labels") else None, config, arch, split,
                         *p[i].astype(np.float32)))
    return pd.DataFrame(recs, columns=["doc_id", "sent_idx", "domain", "gold", "config_name", "arch", "split",
                                       *[f"p_{l}" for l in C.LABELS]])


def main():
    gold, te = C.gold_docs(), C.test_docs()
    step0(gold)
    print("\nStep 1: OOF store")
    frames = []
    for name, (prefix, ep, all_runs) in C.CONFIGS.items():
        p = C.oof_probs(prefix, ep)
        assert set(p) == set(gold), name
        f1, per = C.score(gold, C.argmax(p))
        print(f"  {name:24s} OOF ep{ep:2d} macro-F1 {f1:.4f}  Dissent {per['Dissent']:.3f} PO {per['PrecedentOverruled']:.3f}"
              f"  test: {all_runs or '-'}")
        frames.append(tidy(p, gold, name, "E", "oof"))
        if all_runs:
            frames.append(tidy(C.test_probs([os.path.join(C.ALL, r) for r in all_runs]), te, name, "E", "test"))
    # the 0.5750 file's model (train docs only, dev-selected epoch) -> its test probabilities, under its CV config name
    frames.append(tidy(C.test_probs([os.path.join(C.ROOT, "runs_p3", "e_full_laya_nomarker")]), te,
                       "E_full_mlp_la_1024", "E", "test"))
    w3 = C.mean_probs([C.oof_probs(*C.CONFIGS[n][:2]) for n in C.WIDE3])
    f1, per = C.score(gold, C.argmax(w3))
    print(f"  {'E_wide3 ensemble':24s} OOF macro-F1 {f1:.4f}  Dissent {per['Dissent']:.3f} PO {per['PrecedentOverruled']:.3f}")
    df = pd.concat(frames, ignore_index=True)
    out = os.path.join(C.LD, "probs_store.parquet")
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {len(df)} rows, configs {df.config_name.nunique()}, splits {df.split.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
