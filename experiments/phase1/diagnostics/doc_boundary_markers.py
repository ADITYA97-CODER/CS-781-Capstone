"""How often do training documents contain new-document / court-office markers AFTER their start?

For every training document (data/corpus/clean/<source>.train.parquet, the text that was tokenized
for pretraining) and every marker below, a document counts if the marker occurs at a position after the
first 10% of the document's legal_bpe_32k tokens. Also counted: occurrence anywhere. The 990 local
references (128-word mid-document slices) are counted anywhere.

  python phase1/diagnostics/doc_boundary_markers.py --out outputs/phase1/data_diagnostics
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
P = lambda *a: os.path.join(PROJECT_DIR, *a)
SOURCES = ["supreme_court", "high_courts", "tribunals", "acts"]
MARKERS = {
    "IN THE SUPREME COURT": re.compile(r"IN THE SUPREME COURT"),
    "IN THE HIGH COURT": re.compile(r"IN THE HIGH COURT"),
    "ITEM NO": re.compile(r"ITEM NO"),
    "COURT NO": re.compile(r"COURT NO"),
    "RECORD OF PROCEEDINGS": re.compile(r"RECORD OF PROCEEDINGS"),
    "S U P R E M E": re.compile(r"S U P R E M E"),
    "typist initials (/xx){2,}": re.compile(r"(?:/[a-z]{2}(?![a-z])){2,}"),
    "(signed order)": re.compile(r"\(signed order\)", re.I),
}
FRAC = 0.10
_tok = None


def _init():
    global _tok
    from transformers import PreTrainedTokenizerFast
    _tok = PreTrainedTokenizerFast.from_pretrained(P("artifacts", "tokenizer", "legal_bpe_32k"))


def scan(args):
    """-> per marker: (anywhere, after10) doc counts and up to 20 example records from this chunk."""
    src, rows = args
    any_c, late_c, ex = defaultdict(int), defaultdict(int), defaultdict(list)
    for doc_idx, text in rows:
        hits = {m: [x.start() for x in r.finditer(text)] for m, r in MARKERS.items()}
        hits = {m: v for m, v in hits.items() if v}
        if not hits:
            continue
        offs = _tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        starts = np.array([o[0] for o in offs])
        cut = FRAC * len(starts)
        for m, pos in hits.items():
            any_c[m] += 1
            tok_pos = [int(np.searchsorted(starts, p, side="right")) - 1 for p in pos]
            late = [p for p, t in zip(pos, tok_pos) if t >= cut]
            if late:
                late_c[m] += 1
                if len(ex[m]) < 20:
                    p = late[0]
                    ex[m].append(dict(source=src, doc_idx=int(doc_idx), char=p, rel_pos=round(tok_pos[pos.index(p)] / len(starts), 3),
                                      excerpt=text[max(0, p - 160):p + 160].replace("\n", " ")))
    return src, dict(any_c), dict(late_c), dict(ex)


def chunks(src, size=2000):
    t = pq.read_table(P("data", "corpus", "clean", f"{src}.train.parquet"), columns=["doc_idx", "text"])
    di, tx = t.column("doc_idx").to_pylist(), t.column("text").to_pylist()
    for i in range(0, len(tx), size):
        yield src, list(zip(di[i:i + size], tx[i:i + size]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=P("outputs", "phase1", "data_diagnostics"))
    ap.add_argument("--workers", type=int, default=40)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n_docs = {s: pq.ParquetFile(P("data", "corpus", "clean", f"{s}.train.parquet")).metadata.num_rows for s in SOURCES}
    any_c = {s: defaultdict(int) for s in SOURCES}
    late_c = {s: defaultdict(int) for s in SOURCES}
    ex = defaultdict(list)
    with Pool(a.workers, initializer=_init) as pool:
        for s in SOURCES:
            for src, ac, lc, e in pool.imap_unordered(scan, chunks(s)):
                for m, v in ac.items():
                    any_c[src][m] += v
                for m, v in lc.items():
                    late_c[src][m] += v
                for m, v in e.items():
                    ex[m].extend(v)
            print("scanned", s, n_docs[s], flush=True)

    refs = json.load(open(P("data", "corpus", "eval", "local_ref.json")))
    ref_c = {m: defaultdict(int) for m in MARKERS}
    ref_ex = defaultdict(list)
    for k, v in refs.items():
        for m, r in MARKERS.items():
            x = r.search(v["text"])
            if x:
                ref_c[m][v["source"]] += 1
                ref_ex[m].append(dict(id=k, source=v["source"], excerpt=v["text"][max(0, x.start() - 120):x.start() + 120]))
    n_ref = defaultdict(int)
    for v in refs.values():
        n_ref[v["source"]] += 1

    rng = random.Random(0)
    examples = {m: rng.sample(v, min(5, len(v))) for m, v in ex.items()}
    res = dict(n_docs=n_docs, after10=late_c, anywhere=any_c, refs=ref_c, n_refs=dict(n_ref),
               examples=examples, ref_examples={m: v[:5] for m, v in ref_ex.items()}, markers={m: r.pattern for m, r in MARKERS.items()})
    json.dump(res, open(os.path.join(a.out, "doc_boundary_markers.json"), "w"), indent=1, default=dict)

    L = [f"Training docs containing the marker AFTER the first 10% of their tokens (anywhere in brackets); % of docs.", ""]
    L.append("| marker | " + " | ".join(f"{s} (n={n_docs[s]:,})" for s in SOURCES) + f" | all (n={sum(n_docs.values()):,}) | 990 refs (anywhere) |")
    L.append("|---|" + "---|" * (len(SOURCES) + 2))
    for m in MARKERS:
        cells = []
        for s in SOURCES:
            l, an = late_c[s].get(m, 0), any_c[s].get(m, 0)
            cells.append(f"{l:,} ({100 * l / n_docs[s]:.2f}%) [{an:,}]")
        tl, ta = sum(late_c[s].get(m, 0) for s in SOURCES), sum(any_c[s].get(m, 0) for s in SOURCES)
        rc = sum(ref_c[m].values())
        rcs = ", ".join(f"{s[:5]} {ref_c[m][s]}" for s in SOURCES if ref_c[m][s])
        cells.append(f"{tl:,} ({100 * tl / sum(n_docs.values()):.2f}%) [{ta:,}]")
        cells.append(f"{rc}" + (f" ({rcs})" if rc else ""))
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", f"References per source: {dict(n_ref)}", ""]
    for m in MARKERS:
        L.append(f"\n### {m}  (pattern `{MARKERS[m].pattern}`)")
        for e in examples.get(m, []):
            L.append(f"- [{e['source']} doc {e['doc_idx']}, at {e['rel_pos']:.0%} of the doc] …{e['excerpt']}…")
        for e in ref_ex.get(m, [])[:5]:
            L.append(f"- [REFERENCE id {e['id']}, {e['source']}] …{e['excerpt']}…")
    open(os.path.join(a.out, "doc_boundary_markers.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
