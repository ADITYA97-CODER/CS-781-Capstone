"""Replay pool for masked task SFT: plain pretraining text from the CLEANED corpus, tokenized.

  python phase1/sft/build_replay.py --tokens 60e6 --out data/sft/replay_v1

Random cleaned train documents (data/corpus/cleaned_v1, so run-ons split, initials removed, garbled dropped),
excluding official-test-source documents, sampled per source in the pretraining mix (SC 0.3, HC 0.4, Trib 0.2,
Acts 0.1) until each source has its share of --tokens. Each document is tokenized (legal_bpe_32k) and followed by
<|endoftext|> (id 0), as in data/tokens/*.bin. Output: <source>.bin (uint16) + meta.json. The SFT trainer cuts
random windows from these streams, exactly like pretraining windows.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
P = lambda *a: os.path.join(PROJECT_DIR, *a)
CLEANED = P("data", "corpus", "cleaned_v1")
MIX = {"supreme_court": 0.3, "high_courts": 0.4, "tribunals": 0.2, "acts": 0.1}
EOS = 0
_tok = None


def encode(texts):
    global _tok
    if _tok is None:
        from transformers import PreTrainedTokenizerFast
        _tok = PreTrainedTokenizerFast.from_pretrained(P("artifacts", "tokenizer", "legal_bpe_32k"))
    return [np.array(_tok(t, add_special_tokens=False)["input_ids"] + [EOS], dtype=np.uint16) for t in texts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=60e6)
    ap.add_argument("--out", default=P("data", "sft", "replay_v1"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "meta.json")):
        sys.exit(f"{a.out} exists; refusing to overwrite")
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    meta = {"args": vars(a), "sources": {}}
    with Pool(a.workers) as pool:
        for src, share in MIX.items():
            budget = int(a.tokens * share)
            path = os.path.join(CLEANED, f"{src}.train.parquet")
            t = pq.read_table(path, columns=["n_words", "test_ids"])
            ok = np.flatnonzero(np.array([len(x) == 0 for x in t.column("test_ids").to_pylist()]))
            words = t.column("n_words").to_numpy()[ok]
            order = rng.permutation(len(ok))
            # enough documents for the budget (~1.3 tokens per word), read in one pass
            cum = np.cumsum(words[order]) * 1.3
            n_docs = int(np.searchsorted(cum, budget * 1.2)) + 1
            rows = np.sort(ok[order[:n_docs]])
            texts = pads.dataset(path).take(rows, columns=["text"]).column("text").to_pylist()
            perm = rng.permutation(len(texts))
            texts = [texts[i] for i in perm]
            chunks = [texts[i:i + 256] for i in range(0, len(texts), 256)]
            out, n, used = [], 0, 0
            for enc in pool.imap(encode, chunks):
                for e in enc:
                    if n >= budget:
                        break
                    out.append(e)
                    n += len(e)
                    used += 1
                if n >= budget:
                    break
            arr = np.concatenate(out)
            arr.tofile(os.path.join(a.out, f"{src}.bin"))
            meta["sources"][src] = dict(tokens=int(len(arr)), docs=used)
            print(src, meta["sources"][src], flush=True)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
