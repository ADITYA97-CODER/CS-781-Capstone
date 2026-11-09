"""Report on the LLM specific labels (pilot or full run).

  python phase1/sft/llm_label_report.py --split train --out outputs/phase1/llm_labels/PILOT.md

- throughput (label_log.jsonl) and the time estimate for all pairs (train + val)
- items per pair by type and label
- validation drop rates (verbatim-match failures, evidence failures, stoplist, parse failures / salvaged answers)
- agreement with the regex extractor (phase1/sft/masking.py, regex part only, no NER) on the regex-covered types
  (citations, sections/articles/rules, witnesses, dates, amounts): span precision / recall (char overlap) and the
  supported/unsupported decisions on matched spans, with examples of disagreements
- random pairs printed with TARGET and the LLM labels, for a hand check
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import llm_labels as L  # noqa: E402
from masking import regex_specifics, target_specifics  # noqa: E402

P = lambda *a: os.path.join(PROJECT_DIR, *a)
REGEX_TYPES = {"citation": "citation", "section": "section", "article": "section", "rule": "section",
               "witness": "witness", "date": "date", "amount": "amount"}     # regex type -> LLM type
LLM_REGEX_TYPES = {"citation", "section", "witness", "date", "amount"}


def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def compare(rec, pair):
    """LLM vs regex on the regex-covered types for one pair.
    -> dict(regex_spans, llm_spans, regex_hit, llm_hit, llm_any_hit, decisions=[(regex_sup, llm_label, regex, llm)])"""
    rx = [s for s in target_specifics(pair["prompt"], pair["target"], None) if s["type"] in REGEX_TYPES]
    llm = [(sp, it) for it in rec["items"] if it["type"] in LLM_REGEX_TYPES for sp in it["spans"]]
    llm_all = [(sp, it) for it in rec["items"] for sp in it["spans"]]
    out = dict(regex_spans=len(rx), llm_spans=len(llm), regex_hit=0, regex_hit_any_type=0, llm_hit=0, decisions=[])
    for s in rx:
        m = [(sp, it) for sp, it in llm_all if overlap((s["start"], s["end"]), sp)]
        if any(it["type"] in LLM_REGEX_TYPES for _, it in m):
            out["regex_hit"] += 1
        if m:
            out["regex_hit_any_type"] += 1
            sp, it = m[0]
            out["decisions"].append((s["supported"], it["label"], s, it))
    for sp, it in llm:
        if any(overlap((s["start"], s["end"]), sp) for s in rx):
            out["llm_hit"] += 1
    return out


def ctx(text, a, b, w=60):
    return ("…" if a > w else "") + text[max(0, a - w):a] + "[[" + text[a:b] + "]]" + text[b:b + w] + ("…" if b + w < len(text) else "")


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--labels", default=P("data", "sft", "llm_labels_v1"))
    ap.add_argument("--pairs", default=P("data", "sft", "masked_sft_v2"))
    ap.add_argument("--out", default=P("outputs", "phase1", "llm_labels", "PILOT.md"))
    ap.add_argument("--n-show", type=int, default=30)
    ap.add_argument("--n-disagree", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    recs = L.load_records_dir(a.labels, a.split)
    n_skipped = sum(bool(r.get("skipped_guard")) for r in recs.values())
    recs = {k: r for k, r in recs.items() if not r.get("skipped_guard")}
    pairs = {L.pair_id(r): r for r in torch.load(os.path.join(a.pairs, f"{a.split}.pt"), weights_only=False)}
    n_all = len(pairs) + (1001 if a.split == "train" else 0)
    rng = np.random.default_rng(a.seed)
    md = [f"# LLM specific labels — {a.split}, {len(recs)} pairs (Qwen2.5-7B-Instruct, vLLM, temperature 0, guided JSON)\n"]
    js = {"pairs": len(recs), "skipped_guard": n_skipped}
    md.append(f"Pairs kept away from the LLM by the held-out guard (30-word overlap with a held-out / official test "
              f"prompt, e.g. a quoted statute): {n_skipped} (not counted below).\n")

    # ---------------- throughput
    logs = [json.loads(l) for l in open(os.path.join(a.labels, "label_log.jsonl"))] if os.path.exists(
        os.path.join(a.labels, "label_log.jsonl")) else []
    logs = [g for g in logs if g.get("split") == a.split]
    if logs:
        tot_pairs = sum(g["chunk_pairs"] for g in logs)
        tot_s = sum(g["chunk_s"] for g in logs)
        rate = tot_pairs / tot_s * 60
        js["throughput"] = dict(pairs=tot_pairs, seconds=tot_s, pairs_per_min=rate, est_hours_all=n_all / rate / 60)
        md.append(f"## Throughput\n\n{tot_pairs} pairs in {tot_s / 60:.1f} min of generation (model load excluded) = "
                  f"**{rate:.1f} pairs/min** → all {n_all:,} SFT pairs (train + val) ≈ **{n_all / rate / 60:.1f} h** "
                  f"on this GPU at this batch size. Mean Qwen input {np.mean([r['qwen_in_tokens'] for r in recs.values()]):.0f} "
                  f"tokens, output {np.mean([r['qwen_out_tokens'] for r in recs.values()]):.0f} tokens.\n")

    # ---------------- validation
    C = Counter()
    for r in recs.values():
        C.update(r["counts"])
    n_raw = max(1, C["raw"])
    fails = Counter(parse_fail=sum(not r["parse_ok"] for r in recs.values()),
                    truncated=sum(r["finish_reason"] == "length" for r in recs.values()),
                    salvaged=sum(bool(r.get("salvaged")) and r["parse_ok"] for r in recs.values()))
    sup_claims = sum(1 for r in recs.values() for it in r["items"] if it["orig_label"] == "supported") + 0
    js["validation"] = dict(counts=dict(C), answers=dict(fails))
    md.append("## Validation drop rates\n")
    md.append(f"Answers: {len(recs)}; unparsable {fails['parse_fail']}; cut off at max_tokens (loops) {fails['truncated']}, "
              f"of which {fails['salvaged']} salvaged (items before the loop kept).\n")
    md.append("| raw items | kept | not verbatim in TARGET | only inside a longer word | stoplist | duplicates | bad shape | empty |\n|---|---|---|---|---|---|---|---|")
    md.append(f"| {C['raw']} | {C['kept']} ({C['kept'] / n_raw:.1%}) | {C['not_verbatim']} ({C['not_verbatim'] / n_raw:.1%}) | "
              f"{C['no_word_match']} ({C['no_word_match'] / n_raw:.1%}) | {C['stoplist']} ({C['stoplist'] / n_raw:.1%}) | "
              f"{C['duplicate']} ({C['label_conflict']} with conflicting labels) | {C['bad_item']} | {C['empty']} |\n")
    md.append(f"Evidence failures (supported → relabeled unsupported): **{C['evidence_fail']}** of {sup_claims} "
              f"items the model called supported ({C['evidence_fail'] / max(1, sup_claims):.1%}); in "
              f"{C['evidence_in_target_only']} of them the 'evidence' was copied from TARGET.\n")

    # ---------------- items per pair
    by = Counter()
    for r in recs.values():
        for it in r["items"]:
            by[(it["type"], it["label"])] += 1
    n = len(recs)
    types = sorted({t for t, _ in by}, key=lambda t: -sum(by[(t, l)] for l in L.LABELS))
    md.append("## Items per pair (after validation)\n")
    md.append("| type | supported | inferable | unsupported | total |\n|---|---|---|---|---|")
    for t in types + ["ALL"]:
        row = [sum(by[(tt, l)] for tt in (types if t == "ALL" else [t])) / n for l in L.LABELS]
        md.append(f"| {t} | " + " | ".join(f"{x:.2f}" for x in row) + f" | {sum(row):.2f} |")
    no_spec = np.mean([len(r["items"]) == 0 for r in recs.values()])
    md.append(f"\nPairs with no specific: {no_spec:.1%}.\n")
    js["items_per_pair"] = {f"{t}|{l}": v / n for (t, l), v in by.items()}
    js["no_specifics_share"] = float(no_spec)
    # evidence relatedness: verified evidence shares a word / number with the text?
    rel = Counter()
    for r in recs.values():
        for it in r["items"]:
            if it["label"] == "supported":
                tw = {w for w in re.findall(r"[A-Za-z]{3,}|\d+", it["text"].lower())}
                ew = set(re.findall(r"[A-Za-z]{3,}|\d+", (it["evidence"] or "").lower()))
                rel["related" if tw & ew else "unrelated"] += 1
    md.append(f"Verified evidence that shares no word/number with the specific (likely a wrong 'supported'): "
              f"{rel['unrelated']} of {rel['related'] + rel['unrelated']} supported items.\n")
    js["evidence_relatedness"] = dict(rel)

    # ---------------- regex agreement
    agg = Counter()
    conf = Counter()
    dis = []
    for pid, r in recs.items():
        c = compare(r, pairs[pid])
        for k in ("regex_spans", "llm_spans", "regex_hit", "regex_hit_any_type", "llm_hit"):
            agg[k] += c[k]
        for rsup, llab, s, it in c["decisions"]:
            conf[("regex supported" if rsup else "regex unsupported", llab)] += 1
            masked_rx, masked_llm = not rsup, llab == "unsupported"
            if masked_rx != masked_llm:
                dis.append((pid, rsup, llab, s, it))
    prec = agg["llm_hit"] / max(1, agg["llm_spans"])
    rec_ = agg["regex_hit"] / max(1, agg["regex_spans"])
    rec_any = agg["regex_hit_any_type"] / max(1, agg["regex_spans"])
    js["regex_agreement"] = dict(agg, precision=prec, recall=rec_, recall_any_type=rec_any,
                                 confusion={f"{k[0]}|{k[1]}": v for k, v in conf.items()})
    md.append("## Agreement with the regex extractor (citations, sections/articles/rules, witnesses, dates, amounts)\n")
    md.append(f"Span overlap, regex as reference: LLM spans of these types {agg['llm_spans']}, regex spans {agg['regex_spans']}. "
              f"**Precision {prec:.1%}** (LLM spans overlapping a regex span), **recall {rec_:.1%}** (regex spans overlapped "
              f"by an LLM span of these types; {rec_any:.1%} counting LLM spans of any type). Note the regex is not ground "
              f"truth: LLM spans it lacks include exhibits typed as 'other', case numbers, partial dates.\n")
    tot_m = sum(conf.values())
    md.append("Decisions on matched spans (masked = regex unsupported / LLM unsupported):\n")
    md.append("| | LLM supported | LLM inferable | LLM unsupported |\n|---|---|---|---|")
    for rl in ("regex supported", "regex unsupported"):
        md.append(f"| {rl} | " + " | ".join(str(conf[(rl, l)]) for l in L.LABELS) + " |")
    n_dis = len(dis)
    md.append(f"\nMask decision differs on **{n_dis} of {tot_m} matched spans ({n_dis / max(1, tot_m):.1%})**. "
              f"{min(a.n_disagree, n_dis)} random disagreements:\n")
    for i in rng.permutation(n_dis)[:a.n_disagree]:
        pid, rsup, llab, s, it = dis[i]
        pr = pairs[pid]
        md.append(f"- `{pid}` — regex **{'supported' if rsup else 'unsupported'}** (`{s['key']}`) vs LLM **{llab}** "
                  f"(text `{it['text']}`, type {it['type']}, evidence `{it['evidence']}`)  \n  TARGET: {ctx(pr['target'], s['start'], s['end'])}"
                  + (f"  \n  PROMPT (regex match): {ctx(pr['prompt'], pos, pos + 25, 50)}" if (pos := prompt_pos(pr['prompt'], s['key'])) is not None else ""))

    # ---------------- random pairs
    md.append(f"\n## {a.n_show} random pairs for a hand check\n")
    md.append("Labels: **S** supported (evidence verified in PROMPT), **I** inferable, **U** unsupported (masked in mask_source=llm). "
              "`S→U` = the model said supported but its evidence was not in PROMPT.\n")
    ids = list(recs)
    for i in rng.permutation(len(ids))[:a.n_show]:
        pid = ids[i]
        r, pr = recs[pid], pairs[pid]
        md.append(f"### `{pid}`\n\nTARGET: {pr['target']}\n")
        if not r["items"]:
            md.append("- (no specifics)\n")
        for it in r["items"]:
            lab = {"supported": "S", "inferable": "I", "unsupported": "U"}[it["label"]]
            if it["orig_label"] != it["label"]:
                lab = "S→U"
            ev = f" — evidence: “{it['evidence']}”" if it["evidence"] else ""
            md.append(f"- **{lab}** `{it['text']}` ({it['type']}, ×{len(it['spans'])}){ev}")
        if r["dropped"]:
            md.append("- dropped: " + "; ".join(f"`{d['text'][:60]}` ({d['why']})" for d in r["dropped"][:8]))
        md.append("")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(md))
    json.dump(js, open(a.out.replace(".md", ".json"), "w"), indent=1)
    print("\n".join(md[:60]))


def prompt_pos(prompt, key):
    """char position of the first occurrence of a regex key in the prompt (None if absent)."""
    hit = regex_specifics(prompt).get(key)
    return hit[1][0] if hit else None


if __name__ == "__main__":
    main()
