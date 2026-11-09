"""Write the cleaned corpus (phase1/corpus/clean_document.py applied to every document) to a NEW location.

  python phase1/corpus/write_cleaned_corpus.py            # -> data/corpus/cleaned_v1/

Reads data/corpus/clean/<source>.{train,val}.parquet (never modified) and writes
  data/corpus/cleaned_v1/<source>.{train,val}.parquet   columns: doc_idx (original), part, text, n_words, year, court, test_ids
  data/corpus/cleaned_v1/manifest.json                  counts per source/split, dropped (garbled) doc_idx, split doc_idx
A split document yields rows with the same doc_idx and part = 0, 1, ...; >= 20% Kruti-Dev documents yield no rows.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_DIR)
from phase1.corpus import clean_document as cd  # noqa: E402

P = lambda *a: os.path.join(PROJECT_DIR, *a)
SOURCES = ["supreme_court", "high_courts", "tribunals", "acts"]


def work(args):
    src, rows = args
    out, c, dropped, split_docs = [], Counter(), [], []
    for r in rows:
        text = r["text"]
        c["docs_in"] += 1
        c["words_in"] += r["n_words"]
        if cd.is_garbled(text):
            dropped.append(r["doc_idx"])
            c["docs_dropped_garbled"] += 1
            c["words_dropped_garbled"] += r["n_words"]
            continue
        docs, info = cd.analyze_document(text, src)
        if info["splits"]:
            c["docs_split"] += 1
            c["splits"] += len(info["splits"])
            c["splits_ap_signature"] += sum(s["ap_signature_only"] for s in info["splits"])
            split_docs.append([r["doc_idx"], len(docs)])
        if info["initials"]:
            c["docs_initials_removed"] += 1
        if info["lead_removed"]:
            c["docs_lead_removed"] += 1
            for k, _ in info["lead_removed"]:
                c[f"lead_{k}"] += 1
        if info["kruti"]:
            c["docs_kruti_flagged_kept"] += 1
        for i, d in enumerate(docs):
            n = len(d.split())
            out.append(dict(doc_idx=r["doc_idx"], part=i, text=d, n_words=n, year=r["year"], court=r["court"], test_ids=r["test_ids"]))
            c["docs_out"] += 1
            c["words_out"] += n
    return src, out, c, dropped, split_docs


def chunks(path, src, size=4000):
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=size):
        yield src, b.to_pylist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=P("data", "corpus", "cleaned_v1"))
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "manifest.json")):
        sys.exit(f"{a.out} already has a manifest; refusing to overwrite")
    os.makedirs(a.out, exist_ok=True)
    code = open(cd.__file__, "rb").read()
    manifest = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), input=P("data", "corpus", "clean"),
                    clean_document_md5=hashlib.md5(code).hexdigest(),
                    rules=dict(window_tokens=cd.WINDOW_TOKENS, gap_tokens=cd.GAP_TOKENS, quote_tokens=cd.QUOTE_TOKENS,
                               hc_min_frac=cd.HC_MIN_FRAC, initials_last_frac=cd.INITIALS_LAST_FRAC,
                               lead_chars=cd.LEAD_CHARS, garbled_frac=cd.GARBLED_FRAC,
                               split_sources=["supreme_court", "high_courts"], initials_sources=["supreme_court", "high_courts", "tribunals"]),
                    counts={}, dropped_garbled_doc_idx={}, split_doc_idx={})
    schema = pa.schema([("doc_idx", pa.int64()), ("part", pa.int32()), ("text", pa.large_string()), ("n_words", pa.int64()),
                        ("year", pa.large_string()), ("court", pa.large_string()), ("test_ids", pa.list_(pa.string()))])
    with Pool(a.workers) as pool:
        for split in ["train", "val"]:
            for s in SOURCES:
                path = P("data", "corpus", "clean", f"{s}.{split}.parquet")
                key = f"{s}.{split}"
                c, dropped, split_docs = Counter(), [], []
                with pq.ParquetWriter(os.path.join(a.out, f"{key}.parquet"), schema, compression="zstd") as w:
                    # imap (ordered) keeps the original document order
                    for _, out, cc, dr, sd in pool.imap(work, chunks(path, s)):
                        if out:
                            w.write_table(pa.Table.from_pylist(out, schema=schema))
                        c.update(cc)
                        dropped += dr
                        split_docs += sd
                manifest["counts"][key] = dict(c)
                manifest["dropped_garbled_doc_idx"][key] = dropped
                manifest["split_doc_idx"][key] = split_docs
                print(key, dict(c), flush=True)
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
