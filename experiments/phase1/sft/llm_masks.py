"""Token masks for masked SFT from the LLM specific labels (flag mask_source; label_specifics_llm.py -> llm_labels_v1).

  python phase1/sft/llm_masks.py --mask-source llm      # -> data/sft/llm_labels_v1/masks_llm_train.pt + stats json
  python phase1/sft/llm_masks.py --mask-source union

mask_source (per target token: unsup = masked out of the loss, sup = up-weighted when supported_weight != 1):
  regex   the existing flags stored in the pairs (build_pairs.py / masking.py: regex + spaCy NER)       [no file needed]
  llm     unsup = token overlaps an LLM "unsupported" span; sup = overlaps an LLM "supported" span (and not unsup);
          "inferable" spans keep their loss at weight 1
  union   unsup = (regex unsup OR LLM unsup) AND NOT inside an LLM "supported" span with verified evidence;
          sup   = LLM supported (verified) and not unsup
Spans are character spans in the target (every occurrence of each LLM text, llm_labels.find_spans), mapped to tokens with
the legal_bpe_32k offset mapping exactly as build_pairs.encode does, so every sub-word token of a span is covered.
A pair without an LLM record (not labeled) is an error unless --allow-missing (then it falls back to its regex flags and
is counted). Pairs the held-out guard kept away from the LLM (skipped_guard) always fall back to their regex flags.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import llm_labels as L  # noqa: E402

P = lambda *a: os.path.join(PROJECT_DIR, *a)
MASK_SOURCES = ("regex", "llm", "union")
_tok = None


def tok():
    global _tok
    if _tok is None:
        from transformers import PreTrainedTokenizerFast
        _tok = PreTrainedTokenizerFast.from_pretrained(P("artifacts", "tokenizer", "legal_bpe_32k"))
    return _tok


def target_offsets(target):
    """(ids, offsets into `target`) exactly as build_pairs.encode tokenizes a target (leading space)."""
    te = tok()(" " + target, add_special_tokens=False, return_offsets_mapping=True)
    return te["input_ids"], np.array([(max(0, a - 1), max(0, b - 1)) for a, b in te["offset_mapping"]],
                                     dtype=np.int64).reshape(-1, 2)


def span_hits(offsets, spans):
    """bool per token: overlaps any (start, end) char span."""
    s, e = offsets[:, 0], offsets[:, 1]
    hit = np.zeros(len(offsets), dtype=bool)
    for a, b in spans:
        hit |= (s < b) & (e > a)
    return hit


def llm_hits(offsets, items):
    """-> dict label -> bool per token; "supported_verified" only for supported items with verified evidence."""
    out = {}
    for lab in ("supported", "inferable", "unsupported"):
        out[lab] = span_hits(offsets, [sp for it in items if it["label"] == lab for sp in it["spans"]])
    out["supported_verified"] = span_hits(
        offsets, [sp for it in items if it["label"] == "supported" and it.get("evidence_ok") for sp in it["spans"]])
    return out


def flags(offsets, items, mask_source, regex_unsup, regex_sup):
    """-> (unsup, sup) bool arrays over the target tokens for one pair."""
    regex_unsup, regex_sup = np.asarray(regex_unsup, dtype=bool), np.asarray(regex_sup, dtype=bool)
    if mask_source == "regex":
        return regex_unsup.copy(), regex_sup.copy()
    h = llm_hits(offsets, items)
    if mask_source == "llm":
        uns = h["unsupported"]
    elif mask_source == "union":
        uns = (regex_unsup | h["unsupported"]) & ~h["supported_verified"]
    else:
        raise ValueError(f"mask_source must be one of {MASK_SOURCES}")
    return uns, h["supported_verified" if mask_source == "union" else "supported"] & ~uns


def type_shares(offsets, items, uns):
    """masked tokens attributed to the LLM type of the span(s) covering them (else 'regex_only')."""
    c = Counter()
    covered = np.zeros(len(offsets), dtype=bool)
    for it in items:
        h = span_hits(offsets, it["spans"]) & uns & ~covered
        c[it["type"]] += int(h.sum())
        covered |= h
    c["regex_only"] += int((uns & ~covered).sum())
    return c


def build(pairs, records, mask_source, allow_missing=False):
    """pairs: SFT records (train.pt order); records: {pair_id: LLM record}. -> (out dict, stats)."""
    unsup_l, sup_l, ids = [], [], []
    st = defaultdict(Counter)
    for r in pairs:
        pid = L.pair_id(r)
        rec = records.get(pid)
        g = r["source"]
        skipped = rec is not None and rec.get("skipped_guard")
        if rec is None or skipped:
            if rec is None and not allow_missing:
                raise KeyError(f"no LLM labels for {pid}")
            if skipped:                                # never sent to the LLM (held-out guard): regex flags
                st["all"]["skipped_guard"] += 1; st[g]["skipped_guard"] += 1
            uns, sup = np.asarray(r["unsup"], bool), np.asarray(r["sup"], bool)
            st["all"]["missing"] += 1; st[g]["missing"] += 1
            tc = Counter(regex_only=int(uns.sum()))
        else:
            tids, offs = target_offsets(r["target"])
            assert list(tids) == [int(x) for x in r["target_ids"]], f"{pid}: target tokenization differs from the pairs"
            uns, sup = flags(offs, rec["items"], mask_source, r["unsup"], r["sup"])
            tc = type_shares(offs, rec["items"], uns)
        for k in ("all", g):
            st[k]["pairs"] += 1
            st[k]["target_tokens"] += len(uns)
            st[k]["masked"] += int(uns.sum())
            st[k]["supported"] += int(sup.sum())
            st[k]["regex_masked"] += int(np.asarray(r["unsup"], bool).sum())
            st[k]["both_masked"] += int((uns & np.asarray(r["unsup"], bool)).sum())
            for t, v in tc.items():
                st[k]["masked_type:" + t] += v
        unsup_l.append(np.packbits(uns)); sup_l.append(np.packbits(sup)); ids.append(pid)
    stats = {}
    for k, c in st.items():
        n = max(1, c["target_tokens"])
        stats[k] = dict(pairs=c["pairs"], missing=c["missing"], skipped_guard=c["skipped_guard"], share_masked=c["masked"] / n,
                        share_supported=c["supported"] / n, regex_share_masked=c["regex_masked"] / n,
                        share_masked_by_both=c["both_masked"] / n,
                        masked_share_by_type={t.split(":", 1)[1]: v / n for t, v in sorted(c.items())
                                              if t.startswith("masked_type:")})
    return dict(mask_source=mask_source, pair_ids=ids, unsup=unsup_l, sup=sup_l,
                n_tokens=[len(r["target_ids"]) for r in pairs]), stats


def apply_masks(pairs, mask_file):
    """Override r["unsup"] / r["sup"] of the SFT pairs (train.pt order) with a masks_<source>_<split>.pt file."""
    import torch
    m = torch.load(mask_file, weights_only=False)
    assert m["pair_ids"] == [L.pair_id(r) for r in pairs], f"{mask_file}: pair order differs from the pairs file"
    for r, u, s, n in zip(pairs, m["unsup"], m["sup"], m["n_tokens"]):
        assert n == len(r["target_ids"])
        r["unsup"] = np.unpackbits(u, count=n).astype(bool)
        r["sup"] = np.unpackbits(s, count=n).astype(bool)
    return m["mask_source"]


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-source", required=True, choices=["llm", "union"])
    ap.add_argument("--pairs", default=P("data", "sft", "masked_sft_v2"))
    ap.add_argument("--labels", default=P("data", "sft", "llm_labels_v1"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    pairs = torch.load(os.path.join(a.pairs, f"{a.split}.pt"), weights_only=False)
    records = L.load_records_dir(a.labels, a.split)
    out, stats = build(pairs, records, a.mask_source, a.allow_missing)
    path = os.path.join(a.labels, f"masks_{a.mask_source}_{a.split}.pt")
    if os.path.exists(path):
        sys.exit(f"{path} exists; refusing to overwrite")
    torch.save(out, path)
    json.dump(stats, open(path.replace(".pt", "_stats.json"), "w"), indent=1)
    print(json.dumps(stats["all"], indent=1))


if __name__ == "__main__":
    main()
