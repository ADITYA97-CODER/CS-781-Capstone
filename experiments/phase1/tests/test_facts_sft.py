"""Tests for the facts-block ceiling test.   python phase1/tests/test_facts_sft.py   (CPU only)

1 no leakage       no fact in any block (train/val/held-out) is a specific that appears only in the continuation:
                   every key of every fact occurs in the prompt, or (distractors) not in the continuation either
2 truncation       facts.truncate_prompt keeps the prompt's end; Run F and Run C see identical prompt ids
                   (pair_segment with/without block), and the plans (rows, replay, step order) are identical
3 labels           -100 on every facts-block and prompt token; target labels as in the masked mode
4 held-out docs    no pair comes from a held-out document, a start-SFT (masked_sft_v2) train document, or shares a
                   30-word passage with a held-out prompt/reference
5 block cap        every block <= 150 tokens; generate.py --prefix-facts row = block + prompt end
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_DIR, "phase1"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "phase1", "sft"))
from build_pairs import heldout_shingles, shingles, tok  # noqa: E402
from facts import MAX_BLOCK_TOKENS, MAX_PROMPT_TOKENS, MAX_TOTAL_TOKENS, key_in, truncate_prompt  # noqa: E402
from attention_reach import regex_specifics  # noqa: E402
from train_sft import Plan, load_replay, pair_segment  # noqa: E402

P = lambda *a: os.path.join(PROJECT_DIR, *a)
PAIRS = os.environ.get("FACTS_PAIRS", P("data", "sft", "facts_v1"))
_data = {}


def data(split):
    if split not in _data:
        _data[split] = torch.load(os.path.join(PAIRS, f"{split}.pt"), weights_only=False)
    return _data[split]


def test_no_continuation_only_facts():
    n_facts = n_dist = 0
    for split in ("train", "val"):
        for r in data(split):
            prk, trk = set(regex_specifics(r["prompt"])), set(regex_specifics(r["target"]))
            for f in r["facts"]:
                n_facts += 1
                for k in f["keys"]:
                    in_p, in_t = key_in(k, r["prompt"], prk), key_in(k, r["target"], trk)
                    assert in_p or not in_t, (split, r["doc_idx"], f, "continuation-only specific in block")
                    if f.get("distractor"):
                        assert not in_p and not in_t, (f, "distractor must be absent from prompt and target")
                    else:
                        assert in_p, (f, "non-distractor fact must come from the prompt")
                n_dist += bool(f.get("distractor"))
                if not f.get("distractor") and r["facts_kind"] == "oracle":
                    assert f["used"], f
    prompts = json.load(open(P("data", "corpus", "eval", "local_test.json")))
    refs = json.load(open(P("data", "corpus", "eval", "local_ref.json")))
    for kind in ("oracle", "all"):
        blocks = json.load(open(P("outputs", "phase1", "facts_ceiling", "blocks", f"{kind}.json")))
        for k, b in blocks.items():
            prk = set(regex_specifics(prompts[k]["text"]))
            for f in b["facts"]:
                assert all(key_in(x, prompts[k]["text"], prk) for x in f["keys"]), (k, f)
                if kind == "oracle":
                    rrk = set(regex_specifics(refs[k]["text"]))
                    assert any(key_in(x, refs[k]["text"], rrk) for x in f["keys"]), (k, f)
    print(f"1 no leakage: OK ({n_facts} train/val facts, {n_dist} distractors; held-out oracle/all blocks from prompts)")


def test_truncation():
    ids = list(range(1300))
    for nb, nt in [(0, 160), (150, 160), (40, 400), (6, 165)]:
        kept, dropped = truncate_prompt(ids, nb, nt)
        assert kept == ids[len(ids) - len(kept):] and kept[-1] == ids[-1], "prompt end must be kept"
        assert nb + len(kept) <= MAX_PROMPT_TOKENS and nb + len(kept) + nt <= MAX_TOTAL_TOKENS
        assert dropped == len(ids) - len(kept)
    for r in data("train")[:3000] + data("val"):
        full = tok()(r["prompt"], add_special_tokens=False)["input_ids"]
        p = r["prompt_ids"].tolist()
        assert p == full[len(full) - len(p):], "stored prompt ids must be the end of the prompt"
        assert len(full) - len(p) == r["prompt_trunc"]
        idsF, labF, _, _ = pair_segment(r, "unsupported+upweight", 2.0, facts=True)
        idsC, labC, _, _ = pair_segment(r, "unsupported+upweight", 2.0, facts=False)
        nb = len(r["facts_ids"])
        assert idsF[:nb].tolist() == r["facts_ids"].tolist()
        assert idsF[nb:].tolist() == idsC.tolist(), "Run F and Run C must see identical prompt + target ids"
        assert labF[nb:].tolist() == labC.tolist()
        assert len(idsF) <= 1536
    # identical plans for F and C
    cfgF = json.load(open(P("phase1", "configs", "sft", "archE_sft_facts_F.json")))
    cfgC = json.load(open(P("phase1", "configs", "sft", "archE_sft_facts_C.json")))
    pairs = data("train")[:2000]
    replay = load_replay(P(cfgF["data"]["replay"]))
    pF, pC = Plan(pairs, replay, cfgF, seed=0), Plan(pairs, replay, cfgC, seed=0)
    assert pF.steps == pC.steps and dict(pF.stats) == dict(pC.stats), "F and C plans differ"
    assert pF.facts and not pC.facts
    xF, yF, _, _ = pF.materialize(pF.steps[0], pairs, "unsupported+upweight", 2.0)
    xC, yC, _, _ = pC.materialize(pC.steps[0], pairs, "unsupported+upweight", 2.0)
    assert int((yF != -100).sum()) == int((yC != -100).sum()), "same loss tokens in F and C"
    print(f"2 truncation: OK (prompt end kept, F/C prompt ids identical, identical plans over {len(pF.steps)} steps)")


def test_labels():
    n = 0
    for r in data("train")[:3000]:
        ids, lab, w, _ = pair_segment(r, "unsupported+upweight", 2.0, facts=True)
        nb, npr, nt = len(r["facts_ids"]), len(r["prompt_ids"]), len(r["target_ids"])
        # position j predicts token j+1: block and prompt tokens (positions 0 .. nb+npr-2) are never targets
        assert (lab[:nb + npr - 1] == -100).all(), "label on facts block / prompt"
        assert (w[:nb + npr - 1] == 0).all()
        tl = lab[nb + npr - 1:nb + npr - 1 + nt]
        tgt = r["target_ids"].astype(np.int64)
        assert ((tl == tgt) | (tl == -100)).all()
        assert ((tl == -100) == r["unsup"]).all(), "masked mode: unsupported target tokens -100, others trained"
        assert (lab[nb + npr - 1 + nt:] == -100).all()
        n += 1
    print(f"3 labels: OK ({n} pairs: -100 on block + prompt, target as unsupported+upweight)")


def test_no_heldout_docs():
    ho = json.load(open(P("outputs", "phase1", "masked_sft", "heldout_stats.json")))
    ho_docs = defaultdict(set)
    for r in ho.values():
        ho_docs[r["source"]].add(r["doc_idx"])
    v2 = defaultdict(set)
    for r in torch.load(P("data", "sft", "masked_sft_v2", "train.pt"), weights_only=False):
        v2[r["source"]].add(r["doc_idx"])
    ho_sh = heldout_shingles()
    for split in ("train", "val"):
        for r in data(split):
            if r["split"] == "val" or split == "val":
                assert r["doc_idx"] not in ho_docs[r["source"]], ("held-out doc", r["source"], r["doc_idx"])
            if split == "train":
                assert r["split"] == "train"
                assert r["doc_idx"] not in v2[r["source"]], ("start-SFT doc reused", r["doc_idx"])
            assert not (shingles(r["prompt"].split()) & ho_sh), "30-word overlap with held-out"
    print(f"4 held-out docs: OK ({len(data('train'))} train + {len(data('val'))} val pairs; none held-out, none from masked_sft_v2)")


def test_block_cap_and_generate_rows():
    for split in ("train", "val"):
        assert max(len(r["facts_ids"]) for r in data(split)) <= MAX_BLOCK_TOKENS
    for kind in ("oracle", "all", "empty", "random"):
        b = json.load(open(P("outputs", "phase1", "facts_ceiling", "blocks", f"{kind}.json")))
        assert max(v["n_tokens"] for v in b.values()) <= MAX_BLOCK_TOKENS
    # generate.py --prefix-facts row construction (same expressions as in generate.py)
    pre, enc, max_prompt = [5, 6, 7], list(range(100, 1400)), MAX_PROMPT_TOKENS
    enc = enc[-max_prompt:]
    enc = enc[len(enc) - min(len(enc), max_prompt - len(pre)):]
    L = 900
    row = pre + enc[len(enc) - (L - len(pre)):]
    assert row[:3] == pre and row[-1] == 1399 and len(row) == L and len(pre) + len(enc) <= max_prompt
    print("5 block cap / generate rows: OK")


if __name__ == "__main__":
    test_no_continuation_only_facts()
    test_truncation()
    test_labels()
    test_no_heldout_docs()
    test_block_cap_and_generate_rows()
    print("all tests passed")
