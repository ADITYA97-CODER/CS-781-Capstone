"""Build task-shaped SFT pairs (512-word prompt -> next 128 words) from the CLEANED corpus.

  python phase1/sft/build_pairs.py --n-train 50000 --n-val 1000 --out data/sft/masked_sft_v1

Construction mirrors how the held-out / official prompts were made (phase1/corpus/filter_split.py): a document
with >= 640 words is chosen uniformly, the cut is a uniform word index in [512, n_words - 128], the prompt is
the 512 words before the cut and the target the 128 words after it (whitespace words, joined by single
spaces). Here 1-3 cuts are taken per train document (distinct cut positions; val: 1-6, as val documents are scarce). Source mix = the held-out mix
(360/380/170/80 of 990). This reproduces the prompt-length, mid-sentence, cut-position and source-mix
statistics; phase1/tests/test_masked_sft.py compares them with the 990 held-out prompts.

Documents:
  train pairs: data/corpus/cleaned_v1/<source>.train.parquet; val pairs: <source>.val.parquet (never pretrained on)
  excluded: documents that contain an official test prompt (test_ids non-empty), the val documents behind the
  990 held-out prompts (all parts), and any document sharing a 30-word shingle with a held-out prompt or reference.
  >= 20% Kruti-Dev documents are already absent from cleaned_v1.

Per pair it stores prompt / target token ids (legal_bpe_32k, prompt left-truncated to 1216 tokens as generate.py
does), and for every target token whether it overlaps an unsupported / supported specific (phase1/sft/masking.py),
so the training script can build labels for any mask_mode. Output: {train,val}.pt + stats.json.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import torch

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.dirname(__file__))
from masking import target_specifics, token_flags  # noqa: E402
from heldout_stats import ends_mid_sentence  # noqa: E402

P = lambda *a: os.path.join(PROJECT_DIR, *a)
CLEANED = P("data", "corpus", "cleaned_v1")
SOURCES = ["supreme_court", "high_courts", "tribunals", "acts"]
MIX = {"supreme_court": 360, "high_courts": 380, "tribunals": 170, "acts": 80}   # held-out prompts per source
PROMPT_WORDS, TARGET_WORDS = 512, 128
MAX_PROMPT_TOKENS = 1536 - 320          # generate.py: max_seq_len - max_new_tokens
MAX_TOTAL_TOKENS = 1535                 # + <|endoftext|> separator = seq_len 1536
SHINGLE = 30

_tok = None


def tok():
    global _tok
    if _tok is None:
        from transformers import PreTrainedTokenizerFast
        _tok = PreTrainedTokenizerFast.from_pretrained(P("artifacts", "tokenizer", "legal_bpe_32k"))
    return _tok


def shingles(words, n=SHINGLE):
    return {hashlib.md5(" ".join(words[i:i + n]).encode()).digest()[:8] for i in range(0, max(1, len(words) - n + 1))}


def heldout_shingles():
    prompts = json.load(open(P("data", "corpus", "eval", "local_test.json")))
    refs = json.load(open(P("data", "corpus", "eval", "local_ref.json")))
    out = set()
    for k in prompts:
        out |= shingles(prompts[k]["text"].split()) | shingles(refs[k]["text"].split())
    return out


def eligible(split, src, exclude_doc_idx):
    t = pq.read_table(os.path.join(CLEANED, f"{src}.{split}.parquet"), columns=["doc_idx", "n_words", "test_ids"])
    n = t.column("n_words").to_numpy()
    di = t.column("doc_idx").to_numpy()
    test = np.array([len(x) > 0 for x in t.column("test_ids").to_pylist()])
    ok = (n >= PROMPT_WORDS + TARGET_WORDS) & ~test & ~np.isin(di, list(exclude_doc_idx))
    return np.flatnonzero(ok), int(test.sum())


def sample_pairs(split, src, n_pairs, rng, exclude_doc_idx, ho_shingles, report, max_cuts=3):
    rows, n_test = eligible(split, src, exclude_doc_idx)
    report["eligible_docs"], report["test_source_docs_excluded"] = len(rows), n_test
    order = rng.permutation(rows)
    ds = pads.dataset(os.path.join(CLEANED, f"{src}.{split}.parquet"))
    pairs, pos, rejected_overlap = [], 0, 0
    while len(pairs) < n_pairs and pos < len(order):
        take = order[pos:pos + max(4000, n_pairs)]   # one read covers ~2x the documents needed
        pos += len(take)
        t = ds.take(np.sort(take), columns=["doc_idx", "part", "text"]).to_pylist()
        rng.shuffle(t)
        for r in t:
            words = r["text"].split()
            if shingles(words) & ho_shingles:
                rejected_overlap += 1
                continue
            n = len(words)
            k = int(rng.integers(1, max_cuts + 1))
            span = n - TARGET_WORDS - PROMPT_WORDS + 1
            for cut in sorted(rng.choice(span, size=min(k, span), replace=False) + PROMPT_WORDS):
                pairs.append(dict(source=src, split=split, doc_idx=r["doc_idx"], part=r["part"], doc_words=n,
                                  cut_word=int(cut), rel_cut=float(cut / n),
                                  prompt=" ".join(words[cut - PROMPT_WORDS:cut]),
                                  target=" ".join(words[cut:cut + TARGET_WORDS])))
            if len(pairs) >= n_pairs:
                break
    report["docs_rejected_shingle_overlap"] = rejected_overlap
    report["docs_used"] = len({(p["doc_idx"], p["part"]) for p in pairs})
    return pairs[:n_pairs]


def encode(p):
    """Token ids + per-target-token flags for one pair (NER spans passed in p['ner'])."""
    t = tok()
    pe = t(p["prompt"], add_special_tokens=False)["input_ids"]
    te = t(" " + p["target"], add_special_tokens=False, return_offsets_mapping=True)
    tid = te["input_ids"]
    offs = [(max(0, a - 1), max(0, b - 1)) for a, b in te["offset_mapping"]]   # offsets into p["target"]
    doc = SimpleNamespace(ents=p["ner"]) if p["ner"] is not None else None   # NER spans computed in the parent
    specs = target_specifics(p["prompt"], p["target"], doc)
    uns, sup = token_flags(offs, specs)
    trunc = max(0, len(pe) - MAX_PROMPT_TOKENS, len(pe) + len(tid) - MAX_TOTAL_TOKENS)
    pe = pe[trunc:]
    return dict(prompt_ids=np.array(pe, dtype=np.uint16), target_ids=np.array(tid, dtype=np.uint16),
                unsup=uns, sup=sup, prompt_trunc=trunc,
                n_spec_sup=sum(s["supported"] for s in specs), n_spec_unsup=sum(not s["supported"] for s in specs),
                spec_types_unsup=Counter(s["type"] for s in specs if not s["supported"]),
                spec_types_sup=Counter(s["type"] for s in specs if s["supported"]),
                mid_sentence=ends_mid_sentence(p["prompt"]))


class _Ent:
    """Picklable stand-in for a spaCy entity (what ner_ok / target_specifics read)."""
    __slots__ = ("text", "label_", "start_char", "end_char")

    def __init__(self, e):
        self.text, self.label_, self.start_char, self.end_char = e.text, e.label_, e.start_char, e.end_char


def add_ner(pairs, n_process):
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    for p, doc in zip(pairs, nlp.pipe((p["target"] for p in pairs), batch_size=64, n_process=n_process)):
        p["ner"] = [_Ent(e) for e in doc.ents]


def summarize(recs):
    out = {}
    groups = defaultdict(list)
    for r in recs:
        groups[r["source"]].append(r)
        groups["all"].append(r)
    for g, rs in groups.items():
        pt = np.array([len(r["prompt_ids"]) + r["prompt_trunc"] for r in rs])
        rc = np.array([r["rel_cut"] for r in rs])
        ntt = sum(len(r["target_ids"]) for r in rs)
        out[g] = dict(
            pairs=len(rs), docs=len({(r["doc_idx"], r["part"]) for r in rs}),
            prompt_tokens=dict(mean=float(pt.mean()), p5=float(np.percentile(pt, 5)), p50=float(np.median(pt)),
                               p95=float(np.percentile(pt, 95)), max=int(pt.max())),
            prompt_truncated=int(sum(r["prompt_trunc"] > 0 for r in rs)),
            target_tokens_mean=ntt / len(rs),
            mid_sentence=float(np.mean([r["mid_sentence"] for r in rs])),
            rel_cut_mean=float(rc.mean()), near_end_10=float((rc >= 0.9).mean()), near_end_20=float((rc >= 0.8).mean()),
            target_tokens_unsupported=float(sum(r["unsup"].sum() for r in rs) / ntt),
            target_tokens_supported=float(sum(r["sup"].sum() for r in rs) / ntt),
            specifics_per_target_supported=float(np.mean([r["n_spec_sup"] for r in rs])),
            specifics_per_target_unsupported=float(np.mean([r["n_spec_unsup"] for r in rs])),
            targets_with_no_specifics=float(np.mean([r["n_spec_sup"] + r["n_spec_unsup"] == 0 for r in rs])),
            unsupported_by_type=dict(sum((r["spec_types_unsup"] for r in rs), Counter()).most_common()),
            supported_by_type=dict(sum((r["spec_types_sup"] for r in rs), Counter()).most_common()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=50000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--out", default=P("data", "sft", "masked_sft_v1"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--no-ner", action="store_true")
    ap.add_argument("--reuse-val", default=None, help="copy val.pt from this pair set instead of sampling new val pairs")
    ap.add_argument("--val-max-cuts", type=int, default=6,
                    help="val docs are scarce once the held-out docs are excluded (SC: 124 left): up to this many cuts per doc")
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "stats.json")):
        sys.exit(f"{a.out} exists; refusing to overwrite")
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    ho = json.load(open(P("outputs", "phase1", "masked_sft", "heldout_stats.json")))
    ho_docs = defaultdict(set)
    for r in ho.values():
        ho_docs[r["source"]].add(r["doc_idx"])
    ho_sh = heldout_shingles()
    total = sum(MIX.values())
    report = {"args": vars(a), "build": {}}
    data = {}
    for split, n in [("train", a.n_train), ("val", a.n_val)]:
        if split == "val" and a.reuse_val:
            data["val"] = torch.load(os.path.join(a.reuse_val, "val.pt"), weights_only=False)
            torch.save(data["val"], os.path.join(a.out, "val.pt"))
            report["build"]["val"] = f"copied from {a.reuse_val}"
            continue
        pairs = []
        for src in SOURCES:
            want = round(n * MIX[src] / total)
            rep = {}
            # train docs are never held-out docs (those are val docs); val pairs skip the 990 held-out docs
            excl = ho_docs[src] if split == "val" else set()
            pairs += sample_pairs(split, src, want, rng, excl, ho_sh, rep, max_cuts=3 if split == "train" else a.val_max_cuts)
            report["build"][f"{split}.{src}"] = rep
            print(split, src, want, rep, f"{time.time() - t0:.0f}s", flush=True)
        if a.no_ner:
            for p in pairs:
                p["ner"] = None
        else:
            add_ner(pairs, n_process=a.workers)
            print(split, "NER done", f"{time.time() - t0:.0f}s", flush=True)
        with Pool(a.workers) as pool:
            enc = pool.map(encode, pairs, chunksize=64)
        recs = []
        for p, e in zip(pairs, enc):
            p.pop("ner")
            recs.append({**{k: v for k, v in p.items() if k not in ("prompt", "target")}, **e,
                         "prompt": p["prompt"], "target": p["target"]})
        order = rng.permutation(len(recs))
        recs = [recs[i] for i in order]
        data[split] = recs
        torch.save(recs, os.path.join(a.out, f"{split}.pt"))
        print(split, "encoded", len(recs), f"{time.time() - t0:.0f}s", flush=True)
    report["stats"] = {s: summarize(data[s]) for s in data}
    tr = {(r["source"], r["doc_idx"], r["part"]) for r in data["train"]}
    va = {(r["source"], r["doc_idx"], r["part"]) for r in data["val"]}
    report["train_val_doc_overlap"] = len(tr & va)
    json.dump(report, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(json.dumps(report["stats"]["train"]["all"], indent=1))
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
