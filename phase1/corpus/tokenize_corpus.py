"""Stage 4: tokenize the cleaned corpus into flat token files for pretraining.

For every source and split this writes
  data/tokens/<tokenizer>/<source>.<split>.bin      uint16 token ids, documents joined by <|endoftext|>
  data/tokens/<tokenizer>/<source>.<split>.idx.npy  int64 start offset of every document in the .bin
and a meta.json with token counts. Training code memory-maps the .bin files and
samples windows from each source according to the mix, so changing the mix never
requires re-tokenizing.
"""

import argparse
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
from transformers import PreTrainedTokenizerFast

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLEAN_DIR = os.path.join(PROJECT_DIR, "data", "corpus", "clean")
TOKENIZER_ROOT = os.path.join(PROJECT_DIR, "artifacts", "tokenizer")
TOKENS_ROOT = os.path.join(PROJECT_DIR, "data", "tokens")

SOURCES = ["supreme_court", "acts", "tribunals", "high_courts"]


def tokenize_file(tokenizer, parquet_path, bin_path, idx_path, batch_docs):
    eos = tokenizer.eos_token_id
    offsets, n_tokens = [], 0
    with open(bin_path, "wb") as out:
        for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=batch_docs, columns=["text"]):
            encoded = tokenizer._tokenizer.encode_batch(batch.column("text").to_pylist(), add_special_tokens=False)
            for enc in encoded:
                ids = np.fromiter(enc.ids, dtype=np.uint16, count=len(enc.ids))
                offsets.append(n_tokens)
                out.write(ids.tobytes())
                out.write(np.uint16(eos).tobytes())
                n_tokens += len(ids) + 1
    np.save(idx_path, np.asarray(offsets, dtype=np.int64))
    return n_tokens, len(offsets)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="legal_bpe_32k")
    p.add_argument("--sources", nargs="+", default=SOURCES)
    p.add_argument("--batch-docs", type=int, default=4096)
    args = p.parse_args()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(os.path.join(TOKENIZER_ROOT, args.tokenizer))
    assert len(tokenizer) <= 65535, "uint16 storage needs vocab < 65536"
    out_dir = os.path.join(TOKENS_ROOT, args.tokenizer)
    os.makedirs(out_dir, exist_ok=True)

    meta_path = os.path.join(out_dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta.update({"tokenizer": args.tokenizer, "vocab_size": len(tokenizer),
                 "eos_id": tokenizer.eos_token_id, "dtype": "uint16"})
    for name in args.sources:
        for split in ["train", "val"]:
            t0 = time.time()
            n_tok, n_docs = tokenize_file(
                tokenizer,
                os.path.join(CLEAN_DIR, f"{name}.{split}.parquet"),
                os.path.join(out_dir, f"{name}.{split}.bin"),
                os.path.join(out_dir, f"{name}.{split}.idx.npy"),
                args.batch_docs,
            )
            meta[f"{name}.{split}"] = {"tokens": n_tok, "docs": n_docs}
            print(f"{name}.{split}: {n_tok / 1e6:,.1f}M tokens, {n_docs:,} docs ({time.time() - t0:.0f}s)", flush=True)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
