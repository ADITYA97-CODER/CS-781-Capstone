"""Dry run of phase1/corpus/clean_document.py over the whole training corpus. Writes ONLY a report.

  python phase1/diagnostics/cleaning_report.py --out outputs/phase1/data_cleaning

Reports per source: documents split (and number of new documents), trailing initials removed, Kruti-Dev
flags, rejected split candidates (by reason), plus 10 random examples of each action / rejection
(300 characters around the change).
"""
import argparse
import json
import os
import random
import sys
import zlib
from collections import Counter, defaultdict
from multiprocessing import Pool

import pyarrow.parquet as pq

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_DIR)
from phase1.corpus.clean_document import analyze_document, kruti_frac  # noqa: E402

P = lambda *a: os.path.join(PROJECT_DIR, *a)
SOURCES = ["supreme_court", "high_courts", "tribunals", "acts"]
K_EX = 60  # examples kept per chunk per kind (then sampled down)


def around(text, pos, cut=None, w=150):
    s = text[max(0, pos - w):pos] + " ⟦✂⟧ " + text[pos:pos + w] if cut else text[max(0, pos - w):pos + w]
    return s.replace("\n", " ")


def scan(args):
    src, chunk_id, rows = args
    rng = random.Random(zlib.crc32(f"{src}{chunk_id}".encode()))
    c = Counter()
    reasons = Counter()
    ex = defaultdict(list)  # kind -> reservoir of (key, record)
    for doc_idx, text in rows:
        docs, info = analyze_document(text, src)
        c["docs"] += 1
        if info["splits"]:
            c["docs_split"] += 1
            c["splits"] += len(info["splits"])
            for s in info["splits"]:
                c[f"splits_date_{s['date_kind']}"] += 1
                if s["ap_signature_only"]:
                    c["splits_ap"] += 1
                    ex["split_ap"].append((rng.random(), dict(source=src, doc_idx=doc_idx, marker=s["marker"], rel_pos=s["rel_pos"],
                                                               excerpt=around(text, s["cut"], cut=True))))
            for kind, txt in info["lead_removed"]:
                c[f"lead_{kind}"] += 1
            if info["lead_removed"]:
                c["docs_lead"] += 1
                ex["lead"].append((rng.random(), dict(source=src, doc_idx=doc_idx, removed=" | ".join(t for _, t in info["lead_removed"]),
                                                       excerpt="")))
            for s in info["splits"]:
                ex["split"].append((rng.random(), dict(source=src, doc_idx=doc_idx, marker=s["marker"], rel_pos=s["rel_pos"], date=s["date_kind"],
                                                        excerpt=around(text, s["cut"], cut=True))))
        if info["initials"]:
            c["docs_initials"] += 1
            p = info["initials"]["pos"]
            ex["initials"].append((rng.random(), dict(source=src, doc_idx=doc_idx, removed=info["initials"]["removed"],
                                                       excerpt=text[max(0, p - 280):p] + " ⟦removed: " + text[p:].strip() + "⟧")))
        if info["kruti"]:
            c["docs_kruti"] += 1
            if kruti_frac(text) >= 0.2:
                c["docs_kruti_heavy"] += 1
            ex["kruti"].append((rng.random(), dict(source=src, doc_idx=doc_idx, excerpt=text[len(text) // 2:len(text) // 2 + 300])))
        # a rejected candidate = marker after the first 10% of the document (start-of-document headers are not candidates)
        rejects = [r for r in info["rejects"] if r["pos"] >= 0.10 * len(text)]
        c["rejects_head"] += len(info["rejects"]) - len(rejects)
        for s in info["splits"]:
            c[f"split_pos_{min(9, int(10 * s['rel_pos']))}"] += 1
        if rejects:
            c["docs_with_reject"] += 1
            c["rejects"] += len(rejects)
            for r in rejects:
                key = "+".join(sorted(x.split("(")[0] if not x.startswith("gap_") else "gap" for x in r["reasons"]))
                reasons[key] += 1
                ex["reject"].append((rng.random(), dict(source=src, doc_idx=doc_idx, marker=r["marker"], reasons=r["reasons"],
                                                         rel_pos=round(r["pos"] / len(text), 3), excerpt=around(text, r["pos"], w=200))))
        c["chars_in"] += len(text)
        c["chars_out"] += sum(len(d) for d in docs)
        c["docs_out"] += len(docs)
    ex = {k: [r for _, r in sorted(v, key=lambda t: t[0])[:K_EX]] for k, v in ex.items()}
    return src, c, reasons, ex


def chunks(src, size=4000):
    pf = pq.ParquetFile(P("data", "corpus", "clean", f"{src}.train.parquet"))
    for i, b in enumerate(pf.iter_batches(batch_size=size, columns=["doc_idx", "text"])):
        yield src, i, list(zip(b.column("doc_idx").to_pylist(), b.column("text").to_pylist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=P("outputs", "phase1", "data_cleaning"))
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--n-ex", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tot = {s: Counter() for s in SOURCES}
    reasons = {s: Counter() for s in SOURCES}
    ex = {s: defaultdict(list) for s in SOURCES}
    with Pool(a.workers) as pool:
        for s in SOURCES:
            for src, c, r, e in pool.imap_unordered(scan, chunks(s)):
                tot[src].update(c)
                reasons[src].update(r)
                for k, v in e.items():
                    ex[src][k].extend(v)
            print("scanned", s, dict(tot[s]), flush=True)

    rng = random.Random(0)
    # 10 examples per action pooled over sources, weighted by how often each source triggers the action
    pooled = {}
    for kind in ["split", "initials", "kruti", "reject", "split_ap", "lead"]:
        allx = [x for s in SOURCES for x in ex[s][kind]]
        w = [tot[x["source"]][{"split": "splits", "initials": "docs_initials", "kruti": "docs_kruti", "reject": "rejects", "split_ap": "splits_ap", "lead": "docs_lead"}[kind]]
             / max(1, len(ex[x["source"]][kind])) for x in allx]
        pick = []
        pool_ = list(zip(allx, w))
        while pool_ and len(pick) < a.n_ex:
            i = rng.choices(range(len(pool_)), weights=[q for _, q in pool_])[0]
            pick.append(pool_.pop(i)[0])
        pooled[kind] = pick
    per_source_rejects = {s: rng.sample(ex[s]["reject"], min(5, len(ex[s]["reject"]))) for s in SOURCES}
    per_source_splits = {s: rng.sample(ex[s]["split"], min(5, len(ex[s]["split"]))) for s in SOURCES}

    json.dump(dict(totals={s: dict(tot[s]) for s in SOURCES}, reject_reasons={s: dict(reasons[s]) for s in SOURCES},
                   examples=pooled, per_source_rejects=per_source_rejects, per_source_splits=per_source_splits),
              open(os.path.join(a.out, "cleaning_dry_run.json"), "w"), indent=1)

    pct = lambda n, d: f"{n:,} ({100 * n / max(1, d):.2f}%)"
    L = ["# Cleaning dry run over the training corpus (no output written)", "",
         "| source | docs | docs split | new docs from splits | split date kind (place / Date: label / bare) | trailing initials removed | Kruti-Dev flagged [≥20% of words garbled] | docs with a rejected split candidate | chars removed |",
         "|---|---|---|---|---|---|---|---|---|"]
    T = Counter()
    for s in SOURCES:
        t = tot[s]
        T.update(t)
        L.append(f"| {s} | {t['docs']:,} | {pct(t['docs_split'], t['docs'])} | {t['splits']:,} | {t['splits_date_place']:,} / {t['splits_date_label']:,} / {t['splits_date_bare']:,} | "
                 f"{pct(t['docs_initials'], t['docs'])} | {pct(t['docs_kruti'], t['docs'])} [{t['docs_kruti_heavy']:,}] | {pct(t['docs_with_reject'], t['docs'])} | "
                 f"{t['chars_in'] - t['chars_out']:,} |")
    L.append(f"| **all** | {T['docs']:,} | {pct(T['docs_split'], T['docs'])} | {T['splits']:,} | {T['splits_date_place']:,} / {T['splits_date_label']:,} / {T['splits_date_bare']:,} | "
             f"{pct(T['docs_initials'], T['docs'])} | {pct(T['docs_kruti'], T['docs'])} [{T['docs_kruti_heavy']:,}] | {pct(T['docs_with_reject'], T['docs'])} | {T['chars_in'] - T['chars_out']:,} |")
    L += ["", "Andhra Pradesh signature form (all-caps NAME, J / , CJ; high_courts only): splits that needed it: "
          + ", ".join(f"{s} {tot[s]['splits_ap']:,}" for s in SOURCES)]
    L += ["", "Leading junk removed from split-off documents (first 100 chars): "
          + "; ".join(f"{s}: docs {tot[s]['docs_lead']:,} (page_footer {tot[s]['lead_page_footer']:,}, typist_mark "
                      f"{tot[s]['lead_typist_mark']:,}, stray_punct {tot[s]['lead_stray_punct']:,})" for s in SOURCES[:2])]
    L += ["", "Position of splits within the original document (decile of the marker position):", ""]
    for s in SOURCES:
        L.append(f"- {s}: " + " ".join(f"{10 * d}%:{tot[s][f'split_pos_{d}']}" for d in range(10)))
    L += ["", "Start-of-document markers (first 10%, not counted as candidates): "
          + ", ".join(f"{s} {tot[s]['rejects_head']:,}" for s in SOURCES)]
    L += ["", "Rejected split candidates (marker after the first 10%) by reason (a candidate = first marker of a cluster):", ""]
    for s in SOURCES:
        L.append(f"- {s}: " + ", ".join(f"{k} {v:,}" for k, v in reasons[s].most_common()))
    titles = dict(split_ap="Splits that needed the Andhra Pradesh signature form", lead="Leading junk removed from split-off documents",
                  split="Splits (⟦✂⟧ = cut point)", initials="Trailing initials removed", kruti="Kruti-Dev flagged (no change)",
                  reject="Rejected split candidates (marker present, conditions not met)")
    for kind in ["split", "split_ap", "lead", "initials", "reject", "kruti"]:
        L += ["", f"## {titles[kind]} — {len(pooled[kind])} random examples", ""]
        for i, x in enumerate(pooled[kind], 1):
            extra = ", ".join(f"{k}={x[k]}" for k in ("marker", "rel_pos", "date", "reasons", "removed") if k in x)
            L.append(f"{i}. [{x['source']} doc {x['doc_idx']}{', ' + extra if extra else ''}]  \n   …{x['excerpt']}…")
    L += ["", "## Extra: 5 random splits and rejects per source", ""]
    for s in SOURCES:
        for kind, d in [("split", per_source_splits), ("reject", per_source_rejects)]:
            for x in d[s]:
                extra = ", ".join(f"{k}={x[k]}" for k in ("marker", "rel_pos", "reasons") if k in x)
                L.append(f"- {kind} [{s} doc {x['doc_idx']}, {extra}] …{x['excerpt']}…")
    open(os.path.join(a.out, "cleaning_dry_run.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:20]))


if __name__ == "__main__":
    main()
