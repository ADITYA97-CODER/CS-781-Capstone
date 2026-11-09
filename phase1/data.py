"""Batches for Phase-1 pretraining, sampled from the tokenized corpus by source mix.

Each training example is a random window of seq_len+1 tokens from one source's
token file (documents are separated by <|endoftext|>). The source is drawn with
probability given by the mix, so the mix is a pure training-time setting.
Random start offsets mean windows begin mid-document and mid-sentence, like the
official test prompts. Batches are a deterministic function of (seed, step), so a
resumed run sees exactly the data it would have seen without the interruption.
"""

import json
import os
import queue
import threading

import numpy as np
import torch

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOKENS_ROOT = os.path.join(PROJECT_DIR, "data", "tokens")


class TokenSource:
    def __init__(self, path):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.tokens)


class MixedWindowSampler:
    def __init__(self, tokenizer_name, mix, seq_len, split="train", seed=0):
        token_dir = os.path.join(TOKENS_ROOT, tokenizer_name)
        with open(os.path.join(token_dir, "meta.json")) as f:
            self.meta = json.load(f)
        total = sum(mix.values())
        self.names = [n for n in mix if mix[n] > 0]
        self.weights = np.array([mix[n] / total for n in self.names])
        self.sources = [TokenSource(os.path.join(token_dir, f"{n}.{split}.bin")) for n in self.names]
        self.seq_len, self.seed = seq_len, seed

    def epochs_per_source(self, total_tokens):
        """How many times each source's tokens are seen for a given training budget."""
        return {n: float(w * total_tokens / len(s)) for n, w, s in zip(self.names, self.weights, self.sources)}

    def batch(self, batch_size, step, micro_step=0):
        rng = np.random.default_rng([self.seed, step, micro_step])
        which = rng.choice(len(self.sources), size=batch_size, p=self.weights)
        rows = np.empty((batch_size, self.seq_len + 1), dtype=np.int64)
        for i, s in enumerate(which):
            toks = self.sources[s].tokens
            start = int(rng.integers(0, len(toks) - self.seq_len - 1))
            rows[i] = toks[start:start + self.seq_len + 1]
        rows = torch.from_numpy(rows)
        return rows[:, :-1], rows[:, 1:]


def fixed_val_windows(tokenizer_name, source, seq_len, n_windows, seed=1234):
    """A fixed set of validation windows for one source (same windows at every evaluation)."""
    path = os.path.join(TOKENS_ROOT, tokenizer_name, f"{source}.val.bin")
    toks = np.memmap(path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(toks) - seq_len - 1, size=n_windows)
    rows = torch.from_numpy(np.stack([toks[s:s + seq_len + 1] for s in starts]).astype(np.int64))
    return rows[:, :-1], rows[:, 1:]


def single_doc_val_windows(tokenizer_name, source, seq_len, n_docs, seed=1234, min_len=64, eos_id=0):
    """Validation rows that each hold ONE document from its first token: up to seq_len+1 tokens
    of a random validation document (at least min_len tokens long, <|endoftext|> excluded).
    Shorter documents are padded with <|endoftext|> and their padded labels are -100. Padding only
    follows the document, so causal attention never lets it affect the scored positions; the
    loss is comparable between models with and without doc_mask."""
    token_dir = os.path.join(TOKENS_ROOT, tokenizer_name)
    toks = np.memmap(os.path.join(token_dir, f"{source}.val.bin"), dtype=np.uint16, mode="r")
    starts = np.load(os.path.join(token_dir, f"{source}.val.idx.npy"))
    ends = np.append(starts[1:], len(toks))
    ends = ends - (toks[ends - 1] == eos_id)                    # drop each document's <|endoftext|>
    ok = np.flatnonzero(ends - starts >= min_len)
    pick = np.sort(np.random.default_rng(seed).choice(ok, size=min(n_docs, len(ok)), replace=False))
    x = torch.full((len(pick), seq_len), eos_id, dtype=torch.int64)
    y = torch.full((len(pick), seq_len), -100, dtype=torch.int64)
    for r, i in enumerate(pick):
        doc = torch.from_numpy(toks[starts[i]:min(ends[i], starts[i] + seq_len + 1)].astype(np.int64))
        x[r, :len(doc) - 1], y[r, :len(doc) - 1] = doc[:-1], doc[1:]
    return x, y


class Prefetcher:
    """Builds the micro-batches of upcoming steps in a background thread (in pinned memory), so the
    GPU never waits for the memmap reads. make_step(step) must be deterministic in `step`."""

    def __init__(self, make_step, start, stop, depth=2):
        self.q = queue.Queue(maxsize=depth)
        self.thread = threading.Thread(target=self._run, args=(make_step, start, stop), daemon=True)
        self.thread.start()

    def _run(self, make_step, start, stop):
        for step in range(start, stop):
            self.q.put((step, [(x.pin_memory(), y.pin_memory()) for x, y in make_step(step)]))

    def get(self, step):
        s, batches = self.q.get()
        assert s == step, (s, step)
        return batches


def mask_boundary_labels(x, y, eos_id):
    """doc_mask: never train '<|endoftext|> -> first token of the next document'. Where the input
    token is <|endoftext|>, the label becomes -100 (ignored). <|endoftext|> itself stays a target:
    it is the label of the last token of the document it ends."""
    return y.masked_fill(x == eos_id, -100)


def mask_for_mlm(x, mask_id, vocab_size, rate, special_max_id, generator=None):
    """BERT-style corruption for Arch C's masked batches: choose `rate` of positions;
    80% -> <mask>, 10% -> random token, 10% unchanged. Targets are -100 elsewhere."""
    probs = torch.full(x.shape, rate)
    chosen = torch.bernoulli(probs, generator=generator).bool() & (x > special_max_id)
    targets = torch.where(chosen, x, torch.full_like(x, -100))
    roll = torch.rand(x.shape, generator=generator)
    inputs = x.clone()
    inputs[chosen & (roll < 0.8)] = mask_id
    rand_pos = chosen & (roll >= 0.8) & (roll < 0.9)
    inputs[rand_pos] = torch.randint(special_max_id + 1, vocab_size, (int(rand_pos.sum()),), generator=generator)
    return inputs, targets
