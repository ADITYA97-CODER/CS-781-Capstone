"""Stage 2 of the Phase-1 corpus pipeline: filter, de-duplicate and split.

Reads data/corpus/normalized/<source>.parquet (from normalize.py) and writes
  data/corpus/clean/<source>.train.parquet   documents used for pretraining
  data/corpus/clean/<source>.val.parquet     held-out documents (never trained on)
  data/corpus/eval/local_test.json           test-like prompts: 512 words, same format as the official test
  data/corpus/eval/local_ref.json            the true 128-word continuations of those prompts
  data/corpus/clean/report.json              documents/words removed by each filter, per source

Filters are applied in order and each removed document is attributed to the first
filter it fails, so the report shows exactly what every rule costs.

The pretraining *mix* (how often each source is sampled) is not decided here; it is
a training-time setting, so mixing ablations never require re-running this script.
"""

import argparse
import hashlib
import json
import os
import random
import re
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IN_DIR = os.path.join(PROJECT_DIR, "data", "corpus", "normalized")
OUT_DIR = os.path.join(PROJECT_DIR, "data", "corpus", "clean")
EVAL_DIR = os.path.join(PROJECT_DIR, "data", "corpus", "eval")

# De-duplication keeps the first copy in this order (highest-quality sources first).
SOURCES = ["supreme_court", "acts", "tribunals", "high_courts"]
# Share of official test prompts drawn from each source (measured by normalize.py).
TEST_SOURCE_SHARE = {"supreme_court": 0.36, "high_courts": 0.38, "tribunals": 0.17, "acts": 0.08}

PROMPT_WORDS, CONTINUATION_WORDS = 512, 128
DIGITS = re.compile(r"\d+")


def near_dup_key(text):
    """Hash that ignores case and every number, so templated orders that differ
    only in case numbers, dates or amounts collapse to one key."""
    return hashlib.md5(DIGITS.sub("0", text.lower()).encode()).hexdigest()


def quality_reason(row, args):
    """Name of the first quality filter the document fails, or None if it passes."""
    if row["n_words"] < args.min_words:
        return "too_short"
    if row["nonascii_raw"] > args.max_nonascii:
        return "non_english"
    if row["alpha_frac"] < args.min_alpha:
        return "low_alpha"
    if row["junk_word_frac"] > args.max_junk:
        return "symbol_junk"
    if not args.min_word_len <= row["mean_word_len"] <= args.max_word_len:
        return "odd_word_length"
    if row["n_words"] < args.repetition_max_words and row["unique_ratio"] < args.min_unique:
        return "repetitive"
    return None


def load_source(name):
    table = pq.read_table(os.path.join(IN_DIR, f"{name}.parquet"))
    return table.to_pandas()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-words", type=int, default=50)
    p.add_argument("--max-nonascii", type=float, default=0.10)
    p.add_argument("--min-alpha", type=float, default=0.60)
    p.add_argument("--max-junk", type=float, default=0.20)
    p.add_argument("--min-word-len", type=float, default=3.0)
    p.add_argument("--max-word-len", type=float, default=9.0)
    p.add_argument("--min-unique", type=float, default=0.15)
    p.add_argument("--repetition-max-words", type=int, default=2000,
                   help="only short docs are checked for repetitiveness; long docs naturally repeat words")
    p.add_argument("--near-dedup", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drop-test-sources", action="store_true",
                   help="remove documents that contain an official test prompt from training")
    p.add_argument("--val-words", type=int, default=2_000_000, help="held-out words per source")
    p.add_argument("--eval-prompts", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=48)
    args = p.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    seen_exact, seen_near = set(), set()
    report = {"args": vars(args), "sources": {}}
    eval_pool = {}

    for name in SOURCES:
        df = load_source(name)
        n_docs, n_words = len(df), int(df.n_words.sum())
        reasons = [quality_reason(r, args) for r in df[[
            "n_words", "nonascii_raw", "alpha_frac", "junk_word_frac", "mean_word_len", "unique_ratio"
        ]].to_dict("records")]

        if args.near_dedup:
            with Pool(args.workers) as pool:
                near_keys = pool.map(near_dup_key, df.text.tolist(), chunksize=2000)
        else:
            near_keys = [None] * n_docs

        is_test_source = (df.test_ids.str.len() > 0).to_numpy()
        for i, (h, nk) in enumerate(zip(df.md5, near_keys)):
            if reasons[i] is not None:
                continue
            if h in seen_exact:
                reasons[i] = "exact_duplicate"
            elif nk is not None and nk in seen_near:
                reasons[i] = "near_duplicate"
            elif args.drop_test_sources and is_test_source[i]:
                reasons[i] = "test_source"
            seen_exact.add(h)
            if nk is not None:
                seen_near.add(nk)

        df["reason"] = reasons
        kept = df[df.reason.isna()]

        # Held-out validation: random documents that are not test sources, until val_words is reached.
        candidates = kept.index[~is_test_source[kept.index]].tolist()
        rng.shuffle(candidates)
        val_idx, total = [], 0
        for i in candidates:
            if total >= args.val_words:
                break
            val_idx.append(i)
            total += int(kept.at[i, "n_words"])
        val = kept.loc[val_idx]
        train = kept.drop(index=val_idx)

        cols = ["doc_idx", "text", "n_words", "year", "court", "test_ids"]
        pq.write_table(pa.Table.from_pandas(train[cols], preserve_index=False),
                       os.path.join(OUT_DIR, f"{name}.train.parquet"), compression="zstd")
        pq.write_table(pa.Table.from_pandas(val[cols], preserve_index=False),
                       os.path.join(OUT_DIR, f"{name}.val.parquet"), compression="zstd")
        eval_pool[name] = val[val.n_words >= PROMPT_WORDS + CONTINUATION_WORDS].text.tolist()

        removed = df[df.reason.notna()].groupby("reason").n_words.agg(["count", "sum"])
        report["sources"][name] = {
            "raw_docs": n_docs, "raw_words": n_words,
            "removed": {k: {"docs": int(v["count"]), "words": int(v["sum"])} for k, v in removed.iterrows()},
            "train_docs": len(train), "train_words": int(train.n_words.sum()),
            "val_docs": len(val), "val_words": int(val.n_words.sum()),
            "test_source_docs_in_train": int(is_test_source[train.index].sum()),
        }
        print(name, json.dumps(report["sources"][name]), flush=True)
        del df, kept, train, val

    # Test-like local evaluation set drawn from held-out documents in the test's source proportions.
    prompts, refs, next_id = {}, {}, 1
    for name, share in TEST_SOURCE_SHARE.items():
        pool_docs = eval_pool[name]
        want = round(args.eval_prompts * share)
        for text in rng.sample(pool_docs, min(want, len(pool_docs))):
            words = text.split()
            start = rng.randint(0, len(words) - PROMPT_WORDS - CONTINUATION_WORDS)
            prompts[str(next_id)] = {"text": " ".join(words[start:start + PROMPT_WORDS])}
            refs[str(next_id)] = {"text": " ".join(words[start + PROMPT_WORDS:start + PROMPT_WORDS + CONTINUATION_WORDS]),
                                  "source": name}
            next_id += 1
    with open(os.path.join(EVAL_DIR, "local_test.json"), "w") as f:
        json.dump(prompts, f)
    with open(os.path.join(EVAL_DIR, "local_ref.json"), "w") as f:
        json.dump(refs, f)
    report["local_eval_prompts"] = len(prompts)

    with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"local eval prompts: {len(prompts)}  ->  {EVAL_DIR}")


if __name__ == "__main__":
    main()
