"""How often do training windows cross a document boundary?

  python phase1/diagnostics/boundary_stats.py [--config phase1/configs/archD_4B_docmask.json] [--samples 200000]

Per source and mix-weighted: share of training chunks (seq_len input tokens) containing an
<|endoftext|>, mean boundaries per chunk, and share of documents longer than seq_len tokens.
Uses the document offsets (.idx.npy) written by corpus/tokenize_corpus.py: every document is
followed by one <|endoftext|>. Windows are sampled like data.MixedWindowSampler (uniform start).
"""

import argparse
import json
import os

import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.path.join(PROJECT_DIR, "phase1", "configs", "archD_4B_docmask.json"))
ap.add_argument("--samples", type=int, default=200000, help="random windows per source")
args = ap.parse_args()

d = json.load(open(args.config))["data"]
L = d["seq_len"]
tok_dir = os.path.join(PROJECT_DIR, "data", "tokens", d["tokenizer"])
mix = {k: v for k, v in d["mix"].items() if v > 0}
rng = np.random.default_rng(0)
rows = {}
print(f"seq_len {L}, {args.samples:,} random windows per source\n")
print(f"{'source':14s} {'docs':>10s} {'mean doc len':>13s} {'docs > seq_len':>15s} "
      f"{'chunks with eot':>16s} {'boundaries/chunk':>17s}")
for src in mix:
    starts = np.load(os.path.join(tok_dir, f"{src}.train.idx.npy"))
    n_tok = os.path.getsize(os.path.join(tok_dir, f"{src}.train.bin")) // 2
    eot = np.append(starts[1:], n_tok) - 1                  # position of each document's <|endoftext|>
    doc_len = eot - starts                                  # tokens excluding the <|endoftext|>
    w = rng.integers(0, n_tok - L - 1, size=args.samples)  # window inputs are tokens w .. w+L-1
    n_eot = np.searchsorted(eot, w + L) - np.searchsorted(eot, w)
    rows[src] = dict(docs=len(starts), mean_len=doc_len.mean(), long=(doc_len > L).mean(),
                     with_eot=(n_eot > 0).mean(), per_chunk=n_eot.mean())
    r = rows[src]
    print(f"{src:14s} {r['docs']:>10,d} {r['mean_len']:>13.0f} {r['long']:>15.1%} "
          f"{r['with_eot']:>16.1%} {r['per_chunk']:>17.2f}")
tot = sum(mix.values())
wavg = lambda k: sum(mix[s] / tot * rows[s][k] for s in mix)
print(f"{'training mix':14s} {'':>10s} {'':>13s} {'':>15s} {wavg('with_eot'):>16.1%} {wavg('per_chunk'):>17.2f}")
print(f"\nwith doc_mask, labels ignored per chunk (= boundaries): {wavg('per_chunk') / L:.3%} of labels")
