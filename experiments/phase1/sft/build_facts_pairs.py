"""Pairs with a facts block for the ceiling test (Run F) and its control (Run C, same pairs, no block).

  python phase1/sft/build_facts_pairs.py --n-train 20000 --out data/sft/facts_v1

Train pairs: phase1/sft/build_pairs.py sampling (same filters, held-out exclusion, 30-word-shingle exclusion, source
mix) with seed 1, and additionally WITHOUT the documents behind data/sft/masked_sft_v2 (the start checkpoint's SFT
data), so no pair repeats. Val pairs: the 1,001 val pairs of masked_sft_v1 (prompts/targets/target flags reused).

Per pair (phase1/sft/facts.py): prompt specifics (regex + spaCy NER) with "used" = appears in the target;
block kind oracle 50% / all 30% / empty 20%; noise on oracle/all: drop each fact p=0.15, with p=0.2 add one fact from
another document whose keys occur neither in this prompt nor in this target. Block capped at 150 tokens.
The prompt is left-truncated so that block + prompt fits (facts.truncate_prompt) and these SAME prompt ids are
used by Run C (train_sft.py data.facts=false), so both runs see identical prompt tokens.
Output: {train,val}.pt (records as masked_sft_v2 + facts_ids, facts, facts_kind, ...) + stats.json.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_pairs import (MIX, P, SOURCES, add_ner, encode, heldout_shingles, sample_pairs, tok)  # noqa: E402
from facts import (MIX as KIND_MIX, block_ids, choose_facts, distractor_ok, mark_used, ner_keys,  # noqa: E402
                   prompt_facts, truncate_prompt)
from attention_reach import regex_specifics  # noqa: E402


def facts_worker(p):
    """-> encode() fields (target ids / flags; None for reused val records) + full prompt ids + prompt facts."""
    out = encode(p) if p.get("ner") is not None or p.get("encode") else {}
    out["prompt_full_ids"] = np.array(tok()(p["prompt"], add_special_tokens=False)["input_ids"], dtype=np.uint16)
    out["pfacts"] = mark_used(prompt_facts(p["prompt"], p["prompt_ner"]), p["target"])
    return out


def prompt_ner(pairs, n_process):
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    nlp.max_length = 2_000_000
    for p, doc in zip(pairs, nlp.pipe((p["prompt"] for p in pairs), batch_size=32, n_process=n_process)):
        p["prompt_ner"] = ner_keys(doc)


def attach_blocks(recs, rng):
    """Kind, noisy fact choice, distractor, block ids, shared prompt truncation (in place)."""
    t = tok()
    kinds, probs = zip(*KIND_MIX)
    rk = [(set(regex_specifics(r["prompt"])), set(regex_specifics(r["target"]))) for r in recs]
    for i, r in enumerate(recs):
        kind = kinds[int(rng.choice(len(kinds), p=probs))]
        pool = []
        if kind != "empty":
            for _ in range(20):                     # one candidate distractor from a different document
                j = int(rng.integers(len(recs)))
                o = recs[j]
                if (o["source"], o["doc_idx"], o["part"]) == (r["source"], r["doc_idx"], r["part"]) or not o["pfacts"]:
                    continue
                f = o["pfacts"][int(rng.integers(len(o["pfacts"])))]
                if distractor_ok(f, r["prompt"], rk[i][0], r["target"], rk[i][1]):
                    pool = [{k: f[k] for k in ("text", "keys", "pos")}]
                    break
        chosen = choose_facts(rng, kind, r["pfacts"], pool)
        ids, kept = block_ids(t, chosen)
        pids, dropped = truncate_prompt(r["prompt_full_ids"], len(ids), len(r["target_ids"]))
        r.update(facts_kind=kind, facts=kept, facts_ids=np.array(ids, dtype=np.uint16), prompt_ids=pids,
                 prompt_trunc=int(dropped), facts_capped=len(chosen) - len(kept),
                 n_prompt_facts=len(r["pfacts"]), n_oracle_facts=sum(f["used"] for f in r["pfacts"]))
        r.pop("prompt_full_ids")


def summarize(recs):
    out = {}
    by = defaultdict(list)
    for r in recs:
        by["total"].append(r)
        by["kind_" + r["facts_kind"]].append(r)
    for g, rs in by.items():
        out[g] = dict(
            pairs=len(rs),
            facts_per_block=float(np.mean([len(r["facts"]) for r in rs])),
            block_tokens_mean=float(np.mean([len(r["facts_ids"]) for r in rs])),
            block_tokens_max=int(max(len(r["facts_ids"]) for r in rs)),
            blocks_capped=int(sum(r["facts_capped"] > 0 for r in rs)),
            distractor_share=float(np.mean([any(f.get("distractor") for f in r["facts"]) for r in rs])),
            used_share_in_block=float(sum(f["used"] for r in rs for f in r["facts"]) / max(1, sum(len(r["facts"]) for r in rs))),
            prompt_facts_mean=float(np.mean([r["n_prompt_facts"] for r in rs])),
            oracle_facts_mean=float(np.mean([r["n_oracle_facts"] for r in rs])),
            empty_oracle_share=float(np.mean([r["n_oracle_facts"] == 0 for r in rs])),
            prompt_truncated=int(sum(r["prompt_trunc"] > 0 for r in rs)),
            total_tokens_max=int(max(len(r["facts_ids"]) + len(r["prompt_ids"]) + len(r["target_ids"]) for r in rs)))
    out["by_source"] = dict(Counter(r["source"] for r in recs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--out", default=P("data", "sft", "facts_v1"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--exclude-pairs", default=P("data", "sft", "masked_sft_v2"),
                    help="skip the documents of this pair set's train.pt (the start checkpoint's SFT data)")
    ap.add_argument("--val-from", default=P("data", "sft", "masked_sft_v1"))
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "stats.json")):
        sys.exit(f"{a.out} exists; refusing to overwrite")
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    excl = defaultdict(set)
    for r in torch.load(os.path.join(a.exclude_pairs, "train.pt"), weights_only=False):
        excl[r["source"]].add(r["doc_idx"])
    print("excluded start-SFT docs:", {k: len(v) for k, v in excl.items()}, f"{time.time() - t0:.0f}s", flush=True)
    ho_sh = heldout_shingles()
    total = sum(MIX.values())
    report = {"args": vars(a), "build": {}}
    data = {}

    # train: new pairs
    pairs = []
    for src in SOURCES:
        want = round(a.n_train * MIX[src] / total)
        rep = {}
        pairs += sample_pairs("train", src, want, rng, excl[src], ho_sh, rep, max_cuts=3)
        report["build"][f"train.{src}"] = rep
        print("train", src, want, rep, f"{time.time() - t0:.0f}s", flush=True)
    add_ner(pairs, n_process=a.workers)
    prompt_ner(pairs, a.workers)
    print("NER done", f"{time.time() - t0:.0f}s", flush=True)
    with Pool(a.workers) as pool:
        enc = pool.map(facts_worker, pairs, chunksize=32)
    recs = []
    for p, e in zip(pairs, enc):
        recs.append({**{k: v for k, v in p.items() if k not in ("ner", "prompt_ner")}, **e})
    recs = [recs[i] for i in rng.permutation(len(recs))]
    attach_blocks(recs, rng)
    data["train"] = recs

    # val: the masked_sft_v1 val pairs, new blocks
    val = torch.load(os.path.join(a.val_from, "val.pt"), weights_only=False)
    prompt_ner(val, a.workers)
    with Pool(a.workers) as pool:
        enc = pool.map(facts_worker, [dict(prompt=r["prompt"], target=r["target"], prompt_ner=r["prompt_ner"]) for r in val],
                       chunksize=32)
    for r, e in zip(val, enc):
        r.pop("prompt_ner")
        r.update(e)
    attach_blocks(val, rng)
    data["val"] = val
    report["build"]["val"] = f"pairs of {a.val_from}/val.pt, new facts blocks"

    for split in ("train", "val"):
        for r in data[split]:
            r.pop("pfacts", None)
        torch.save(data[split], os.path.join(a.out, f"{split}.pt"))
    report["stats"] = {s: summarize(data[s]) for s in data}
    json.dump(report, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(json.dumps(report["stats"], indent=1))
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
