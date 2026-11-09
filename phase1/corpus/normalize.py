"""Stage 1 of the Phase-1 corpus pipeline: normalize every raw document once.

For each source file this script
  * folds text to ASCII and collapses all whitespace to single spaces, matching
    the format of the official Phase-1 test prompts (pure ASCII, no newlines);
  * computes per-document quality statistics used later for filtering;
  * hashes the normalized text for exact de-duplication;
  * flags documents that contain any 10-word window of an official test prompt.

Output: data/corpus/normalized/<source>.parquet (one row per raw document).
No document is dropped here; filtering decisions are made in filter_mix.py.

Usage:
  python phase1/corpus/normalize.py                   # all four sources
  python phase1/corpus/normalize.py --sources acts    # a subset
"""

import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
from collections import Counter
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(PROJECT_DIR, "data", "Phase_1_Pretraining")
OUT_DIR = os.path.join(PROJECT_DIR, "data", "corpus", "normalized")

SOURCES = ["acts", "supreme_court", "high_courts", "tribunals"]
SHINGLE = 10  # words per window for test-overlap detection
MIN_SHARED = 100  # windows shared with one prompt to count as its source document

# Characters seen in the corpus mapped the way the test prompts render them.
ASCII_MAP = str.maketrans({
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "´": "'",
    "–": "-", "—": "-", "―": "-", "‐": "-", "‑": "-", "‒": "-", "−": "-",
    "…": "...", " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "­": "", "​": "", "‌": "", "‍": "", "﻿": "", "‪": "", "‫": "",
    "‬": "", "‭": "", "‮": "",
    "×": "x", "½": "1/2", "¼": "1/4", "¾": "3/4", "₹": "Rs.",
})
SCHEMA = pa.schema([
    ("doc_idx", pa.int64()), ("text", pa.string()), ("n_words", pa.int64()), ("n_chars", pa.int64()),
    ("nonascii_raw", pa.float64()), ("alpha_frac", pa.float64()), ("digit_frac", pa.float64()),
    ("junk_word_frac", pa.float64()), ("mean_word_len", pa.float64()), ("unique_ratio", pa.float64()),
    ("md5", pa.string()), ("test_ids", pa.list_(pa.string())), ("test_max_shared", pa.int64()),
    ("year", pa.string()), ("court", pa.string()),
])
NON_ASCII = re.compile(r"[^\x20-\x7e]")
WORD_HAS_ALNUM = re.compile(r"[A-Za-z0-9]")


def to_ascii(text):
    text = text.translate(ASCII_MAP)
    # Strip accents (e.g. "à" -> "a"), then drop anything still outside printable ASCII.
    text = unicodedata.normalize("NFKD", text)
    text = NON_ASCII.sub(" ", text)
    return " ".join(text.split())


def shingles(words):
    return {" ".join(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def load_test_shingles():
    with open(os.path.join(RAW_DIR, "Test", "test.json")) as f:
        test = json.load(f)
    table = {}
    for key, record in test.items():
        for s in shingles(record["text"].split()):
            table[hash(s)] = key
    return table


# Globals shared with forked workers (copy-on-write).
DOCS = None
TEST_SHINGLES = None


def process_range(bounds):
    start, end = bounds
    rows = []
    for idx in range(start, end):
        doc = DOCS[idx]
        raw = doc.get("text") or ""
        n_raw = max(1, len(raw))
        nonascii_raw = sum(1 for c in raw if ord(c) > 127) / n_raw

        text = to_ascii(raw)
        words = text.split()
        n_words = len(words)
        n_chars = max(1, len(text))
        alpha = sum(c.isalpha() for c in text) / n_chars
        digit = sum(c.isdigit() for c in text) / n_chars
        junk_words = sum(1 for w in words if not WORD_HAS_ALNUM.search(w)) / max(1, n_words)
        mean_word_len = sum(map(len, words)) / max(1, n_words)
        unique_ratio = len(set(words)) / max(1, n_words)

        # A true source document shares ~500 windows with its prompt; generic
        # legal boilerplate shares only a handful, so require MIN_SHARED.
        shared = Counter()
        for i in range(n_words - SHINGLE + 1):
            key = TEST_SHINGLES.get(hash(" ".join(words[i:i + SHINGLE])))
            if key is not None:
                shared[key] += 1
        test_ids = [k for k, c in shared.items() if c >= MIN_SHARED]

        rows.append({
            "doc_idx": idx,
            "text": text,
            "n_words": n_words,
            "n_chars": len(text),
            "nonascii_raw": round(nonascii_raw, 5),
            "alpha_frac": round(alpha, 4),
            "digit_frac": round(digit, 4),
            "junk_word_frac": round(junk_words, 4),
            "mean_word_len": round(mean_word_len, 3),
            "unique_ratio": round(unique_ratio, 4),
            "md5": hashlib.md5(text.lower().encode()).hexdigest(),
            "test_ids": sorted(test_ids, key=int),
            "test_max_shared": max(shared.values(), default=0),
            "year": str(doc.get("year") or doc.get("date") or "")[:4],
            "court": str(doc.get("court") or doc.get("tribunal") or doc.get("category") or ""),
        })
    return rows


def normalize_source(name, workers):
    global DOCS
    t0 = time.time()
    with open(os.path.join(RAW_DIR, "Train", f"train_{name}.json")) as f:
        DOCS = json.load(f)
    print(f"[{name}] loaded {len(DOCS):,} docs in {time.time() - t0:.0f}s", flush=True)

    step = max(1, len(DOCS) // (workers * 8))
    ranges = [(a, min(a + step, len(DOCS))) for a in range(0, len(DOCS), step)]
    out_path = os.path.join(OUT_DIR, f"{name}.parquet")
    writer = pq.ParquetWriter(out_path, SCHEMA, compression="zstd")
    n_rows = 0
    with Pool(workers) as pool:
        for rows in pool.imap(process_range, ranges):
            writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
            n_rows += len(rows)
    writer.close()
    DOCS = None
    print(f"[{name}] wrote {n_rows:,} rows -> {out_path} ({time.time() - t0:.0f}s total)", flush=True)


def main():
    global TEST_SHINGLES
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=SOURCES, choices=SOURCES)
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    TEST_SHINGLES = load_test_shingles()
    print(f"test shingles: {len(TEST_SHINGLES):,}", flush=True)
    for name in args.sources:
        normalize_source(name, args.workers)


if __name__ == "__main__":
    main()
