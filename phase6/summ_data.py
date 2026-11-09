"""Phase 6 (abstractive summarization) data: splits, token cache, document views and packed training rows.

Splits: `train` = Train/train.json minus `dev` (300 docs drawn with seed 0, held out for selection and decoding tuning);
`test` = Test/test.json (100 docs, no summary).

A training example is one row segment, laid out for a decoder-only model:
    [document view tokens] [prompt tokens] [summary tokens] [<|endoftext|>]
The prompt is "\n\nSummary:\n" or, with --len_tag, "\n\nSummary (about N words):\n" with N the reference length rounded to
50 words (at inference N comes from a length predictor fitted on train). The loss is on the summary tokens and the final
<|endoftext|> only. Summaries longer than sum_len tokens are cut and get no end token. Several examples are packed into
one row; the <|endoftext|> that ends each one gives document masking (attention stays inside an example, RoPE restarts),
the padding is <|endoftext|> too (every padding token is its own one-token document, label -100).
"""

import json
import os

import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "Phase_6_Summ")
CACHE = os.path.join(ROOT, "runs_p6", "cache")
TOKENIZER = os.path.join(ROOT, "artifacts", "tokenizer", "legal_bpe_32k", "tokenizer.json")
EOS = 0
N_DEV = 300

_tok = None


def tokenizer():
    global _tok
    if _tok is None:
        _tok = Tokenizer.from_file(TOKENIZER)
    return _tok


def load_split(name):
    """-> list of {"id", "document", "summary"} (summary None for test)."""
    if name == "test":
        return [{"id": r["id"], "document": r["document"], "summary": None}
                for r in json.load(open(os.path.join(DATA, "Test", "test.json")))]
    rows = json.load(open(os.path.join(DATA, "Train", "train.json")))
    dev = set(np.random.RandomState(0).permutation(len(rows))[:N_DEV].tolist())
    keep = [r for i, r in enumerate(rows) if (i in dev) == (name == "dev")]
    assert name in ("train", "dev")
    return keep


def load_tokens(name):
    """-> (ids, doc token arrays, summary token arrays or None, summary word counts). Cached in runs_p6/cache."""
    path = os.path.join(CACHE, f"{name}.npz")
    if not os.path.exists(path):
        rows = load_split(name)
        tok = tokenizer()
        d = tok.encode_batch([r["document"] for r in rows])
        s = tok.encode_batch([r["summary"] or "" for r in rows])
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, ids=np.array([r["id"] for r in rows]),
                 dlens=np.array([len(e.ids) for e in d]), slens=np.array([len(e.ids) for e in s]),
                 dflat=np.concatenate([np.asarray(e.ids, dtype=np.uint16) for e in d]),
                 sflat=np.concatenate([np.asarray(e.ids, dtype=np.uint16) for e in s] + [np.zeros(0, np.uint16)]),
                 swords=np.array([len((r["summary"] or "").split()) for r in rows]),
                 dwords=np.array([len(r["document"].split()) for r in rows]))
    z = np.load(path)
    docs = np.split(z["dflat"].astype(np.int64), np.cumsum(z["dlens"])[:-1])
    sums = np.split(z["sflat"].astype(np.int64), np.cumsum(z["slens"])[:-1]) if name != "test" else None
    return [str(i) for i in z["ids"]], docs, sums, z["swords"], z["dwords"]


def references(name):
    return {r["id"]: r["summary"] for r in load_split(name)}


# ----------------------------------------------------------------------------- views

def doc_view(t, view, budget, head):
    """Which tokens of a document the model reads (at most `budget`).
    head:     the first `budget` tokens.
    headtail: the first `head` tokens + the last budget - head (whole doc if it fits).
    spread:   `n` evenly spaced 512-token pieces covering the document start to end (the first piece is always the
              start of the document, the last its end)."""
    if len(t) <= budget:
        return t
    if view == "head":
        return t[:budget]
    if view == "headtail":
        return np.concatenate([t[:head], t[len(t) - (budget - head):]])
    if view == "spread":
        piece = 512
        n = budget // piece
        starts = np.linspace(0, len(t) - piece, n).round().astype(int)
        return np.concatenate([t[s:s + piece] for s in starts])
    raise ValueError(view)


def length_bucket(words, step=50):
    return int(max(step, round(words / step) * step))


def prompt_tokens(n_words=None):
    text = "\n\nSummary:\n" if n_words is None else f"\n\nSummary (about {n_words} words):\n"
    return np.asarray(tokenizer().encode(text).ids, dtype=np.int64)


class LengthModel:
    """log(summary words) = a + b log(document words), least squares on train."""

    def __init__(self, dwords, swords):
        x, y = np.log(np.maximum(dwords, 1)), np.log(np.maximum(swords, 1))
        self.b, self.a = np.polyfit(x, y, 1)

    def __call__(self, dwords, scale=1.0):
        return np.exp(self.a + self.b * np.log(np.maximum(dwords, 1))) * scale


def build_examples(docs, sums, swords, args):
    """-> list of (input ids, labels) numpy arrays, one per document."""
    out = []
    for i, d in enumerate(docs):
        p = prompt_tokens(length_bucket(swords[i]) if args.len_tag else None)
        v = doc_view(d, args.view, args.doc_len, args.head_len)
        s = sums[i][:args.sum_len]
        end = EOS if len(sums[i]) <= args.sum_len else -100     # a cut summary is not taught to end
        ids = np.concatenate([v, p, s, [EOS]])                  # the input always ends with <|endoftext|> (doc mask)
        lab = np.concatenate([np.full(len(v) + len(p) - 1, -100), s, [end, -100]])  # lab[j] = target of position j
        out.append((ids, lab))
    return out


def pack(examples, order, rows, seq):
    """First-fit packing of examples (in `order`) into micro-batches of `rows` rows of `seq` tokens.
    -> list of micro-batches, each a list of rows, each a list of example indices."""
    mbs, cur, free = [], [], []
    for i in order:
        n = len(examples[i][0])
        assert n <= seq, (n, seq)
        for r, f in enumerate(free):
            if f >= n:
                cur[r].append(i)
                free[r] -= n
                break
        else:
            if len(cur) == rows:
                mbs.append(cur)
                cur, free = [], []
            cur.append([i])
            free.append(seq - n)
    if cur:
        mbs.append(cur)
    return mbs


def collate(mb, examples, rows, seq, device):
    ids = torch.full((rows, seq), EOS, dtype=torch.long)
    lab = torch.full((rows, seq), -100, dtype=torch.long)
    for r, row in enumerate(mb):
        o = 0
        for i in row:
            x, y = examples[i]
            ids[r, o:o + len(x)] = torch.from_numpy(x)
            lab[r, o:o + len(x)] = torch.from_numpy(y)
            o += len(x)
    return ids.to(device, non_blocking=True), lab.to(device, non_blocking=True)
