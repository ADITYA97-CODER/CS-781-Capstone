"""LSI data: splits, token cache and multi-label targets. Document views, packing: phase4/cjpe_data.py (reused).

Splits (EXPERIMENTS.md P5.1):
  train  42,750 docs of Train/train.json
  dev    10,181 docs of Train/dev.json (selection, per-label thresholds; ids disjoint from train and test)
  test   13,019 docs of Test/test.json
Labels: 100 statute ids 0..99; the json stores them as a string "[69, 9, 3]".
"""

import json
import os
import sys

import numpy as np
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phase4"))
from cjpe_data import doc_chunks, micro_batches, pack, TOKENIZER  # noqa: E402,F401

DATA = os.path.join(ROOT, "data", "Phase_5_LSI")
CACHE = os.path.join(ROOT, "runs_p5", "cache")
N_LABELS = 100


def _labels(x):
    x = json.loads(x) if isinstance(x, str) else x
    return sorted({int(v) for v in x})


def load_split(name):
    """-> list of {"id", "text", "labels"} (labels None for test)."""
    path = {"train": "Train/train.json", "dev": "Train/dev.json", "test": "Test/test.json"}[name]
    rows = json.load(open(os.path.join(DATA, path)))
    return [{"id": str(r["id"]), "text": r["text"], "labels": _labels(r["labels"]) if "labels" in r else None}
            for r in rows]


def load_tokens(name):
    """-> (ids list, Y uint8 (n, 100) (all zeros for test), token arrays list of np.int32). Cached in runs_p5/cache."""
    path = os.path.join(CACHE, f"{name}.npz")
    if not os.path.exists(path):
        rows = load_split(name)
        tok = Tokenizer.from_file(TOKENIZER)
        enc = tok.encode_batch([r["text"] for r in rows])
        lens = np.array([len(e.ids) for e in enc], dtype=np.int64)
        flat = np.concatenate([np.asarray(e.ids, dtype=np.uint16) for e in enc])
        Y = np.zeros((len(rows), N_LABELS), dtype=np.uint8)
        for i, r in enumerate(rows):
            Y[i, r["labels"] or []] = 1
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, ids=np.array([r["id"] for r in rows]), Y=Y, lens=lens, flat=flat)
    z = np.load(path)
    offs = np.concatenate([[0], np.cumsum(z["lens"])])
    flat = z["flat"].astype(np.int32)
    toks = [flat[offs[i]:offs[i + 1]] for i in range(len(z["lens"]))]
    return [str(i) for i in z["ids"]], z["Y"], toks


def write_truth(name, path):
    """Truth file for the official evaluator ([{id, gold}]) of a labelled split."""
    json.dump([{"id": r["id"], "gold": r["labels"]} for r in load_split(name)], open(path, "w"))
