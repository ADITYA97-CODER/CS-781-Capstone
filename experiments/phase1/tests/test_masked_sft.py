"""Tests for masked task SFT (phase1/sft/).   python phase1/tests/test_masked_sft.py   (CPU only)

1 masking        unsupported specifics' tokens -> -100 (all subword tokens of a multi-token citation), supported
                 ones kept, every prompt token -100
2 normalization  "Sec. 302" in the target with "Section 302" in the prompt is supported (also "PW 4" vs "PW-4")
3 no leakage     no SFT pair (train or val) comes from a held-out document or shares a 30-word passage with a
                 held-out prompt/reference; train and val documents are disjoint
4 distribution   SFT pairs vs the 990 held-out prompts: prompt length, mid-sentence share, near-end share, source mix
5 loss           step loss == manual mean over unmasked target tokens; supported_weight 1.0 makes
                 "unsupported+upweight" == "unsupported"
6 replay ratio   replay share of loss tokens over 100 batches == config (and the input-token basis)
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_DIR, "phase1"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "phase1", "sft"))
from build_pairs import SHINGLE, encode, heldout_shingles, shingles  # noqa: E402
from masking import target_specifics  # noqa: E402
from train_sft import Plan, load_replay, pair_segment, step_loss  # noqa: E402

P = lambda *a: os.path.join(PROJECT_DIR, *a)
PAIRS = os.environ.get("SFT_PAIRS", P("data", "sft", "masked_sft_v1"))
_nlp = None


def nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    return _nlp


def make_record(prompt, target, ner=True):
    from build_pairs import _Ent
    p = dict(prompt=prompt, target=target, ner=[_Ent(e) for e in nlp()(target).ents] if ner else None)
    return encode(p)


def tokens_of(rec, target, span_text):
    """indices of the target tokens overlapping the first occurrence of span_text in target."""
    from build_pairs import tok
    a = target.index(span_text)
    b = a + len(span_text)
    offs = tok()(" " + target, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    offs = [(max(0, x - 1), max(0, y - 1)) for x, y in offs]
    return [i for i, (x, y) in enumerate(offs) if x < b and y > a]


def test_1_masking():
    prompt = ("The appellant was convicted under Section 302 of the Indian Penal Code. PW-4 deposed that the accused "
              "Ram Kumar was present at the spot on the night of the incident and the trial court relied on")
    target = ("his testimony. Reliance was placed on (2019) 4 SCC 123 and on the judgment dated 12th March, 2019, "
              "and the conviction under Section 302 was upheld for Ram Kumar while Mohan Lal Sharma was acquitted.")
    rec = make_record(prompt, target)
    for mode in ["unsupported", "unsupported+upweight", "none"]:
        ids, labels, weights, _ = pair_segment(rec, mode, 2.0)
        P_, T = len(rec["prompt_ids"]), len(rec["target_ids"])
        tl = labels[P_ - 1:P_ - 1 + T]
        assert (labels[:P_ - 1] == -100).all() and (labels[P_ - 1 + T:] == -100).all(), "prompt / eos labels must be -100"
        cit = tokens_of(rec, target, "(2019) 4 SCC 123")
        date = tokens_of(rec, target, "12th March, 2019")
        sec = tokens_of(rec, target, "Section 302")
        name = tokens_of(rec, target, "Ram Kumar")
        assert len(cit) >= 4, cit
        if mode == "none":
            assert (tl != -100).all()
            continue
        assert (tl[cit] == -100).all() and (tl[date] == -100).all(), "unsupported citation/date not fully masked"
        num302 = tokens_of(rec, target, "302")
        assert (tl[num302] != -100).all() and (tl[name] != -100).all(), "supported specifics must stay"
        assert rec["unsup"][tokens_of(rec, target, "Mohan Lal Sharma")].all(), "unsupported NER name not masked"
        n_masked = int((tl == -100).sum())
        assert n_masked == int(rec["unsup"].sum()) and n_masked < T
        if mode == "unsupported+upweight":
            w = weights[P_ - 1:P_ - 1 + T]
            assert (w[num302] == 2.0).all() and (w[cit] == 0).all() and set(np.unique(w)) <= {0.0, 1.0, 2.0}
    print(f"ok 1 masking: citation {len(cit)} tokens fully masked, date masked, Section 302 / Ram Kumar kept, "
          f"prompt tokens -100; masked {n_masked}/{T} target tokens")


def test_2_normalization():
    for p, t in [("as held under Section 302 IPC the court", "convicted under Sec. 302 IPC and sentenced"),
                 ("charged u/s 302 IPC", "the offence under Sections 302 and 34 IPC"),
                 ("statement of PW-4 was recorded", "the evidence of PW 4 shows"),
                 ("PW 4 Ramesh stated", "P.W.4 is the son of"),
                 ("decided on 12.03.2019", "the order dated 12th March, 2019")]:
        sp = target_specifics(p, t)
        first = [s for s in sp if s["type"] in ("section", "witness", "date")][0]
        assert first["supported"], (p, t, sp)
    sp = target_specifics("as held under Section 302 IPC", "under Sections 302 and 307 IPC")
    assert {s["key"]: s["supported"] for s in sp} == {"sec:302": True, "sec:307": False}, sp
    print("ok 2 normalization: Sec./Section/u/s, PW 4/PW-4/P.W.4, 12.03.2019/12th March, 2019 match; "
          "302 supported and 307 unsupported inside one list")


def test_3_no_leakage():
    ho = json.load(open(P("outputs", "phase1", "masked_sft", "heldout_stats.json")))
    ho_docs = {(r["source"], r["doc_idx"]) for r in ho.values()}
    ho_sh = heldout_shingles()
    n = 0
    docs = {}
    for split in ["train", "val"]:
        recs = torch.load(os.path.join(PAIRS, f"{split}.pt"), weights_only=False)
        docs[split] = {(r["source"], r["doc_idx"], r["part"]) for r in recs}
        for r in recs:
            assert (r["source"], r["doc_idx"]) not in ho_docs, ("held-out doc", r["source"], r["doc_idx"])
            assert not (shingles(r["prompt"].split()) | shingles(r["target"].split())) & ho_sh, "shares a held-out passage"
            n += 1
    assert not docs["train"] & docs["val"], "train / val documents overlap"
    stats = json.load(open(os.path.join(PAIRS, "stats.json")))
    print(f"ok 3 no leakage: {n} pairs; none from the {len(ho_docs)} held-out docs; no {SHINGLE}-word passage shared with "
          f"held-out prompts/refs; train/val docs disjoint; build report: {json.dumps(stats['build'])}")


def test_4_distribution():
    ho = list(json.load(open(P("outputs", "phase1", "masked_sft", "heldout_stats.json"))).values())
    st = json.load(open(os.path.join(PAIRS, "stats.json")))["stats"]
    hpt = np.array([r["prompt_tokens"] for r in ho])
    hrc = np.array([r["rel_cut"] for r in ho])
    hmix = Counter(r["source"] for r in ho)
    rows = [("prompt tokens mean", hpt.mean(), st["train"]["all"]["prompt_tokens"]["mean"], st["val"]["all"]["prompt_tokens"]["mean"]),
            ("prompt tokens p5", np.percentile(hpt, 5), st["train"]["all"]["prompt_tokens"]["p5"], st["val"]["all"]["prompt_tokens"]["p5"]),
            ("prompt tokens p95", np.percentile(hpt, 95), st["train"]["all"]["prompt_tokens"]["p95"], st["val"]["all"]["prompt_tokens"]["p95"]),
            ("mid-sentence share", np.mean([r["mid_sentence"] for r in ho]), st["train"]["all"]["mid_sentence"], st["val"]["all"]["mid_sentence"]),
            ("relative cut mean", hrc.mean(), st["train"]["all"]["rel_cut_mean"], st["val"]["all"]["rel_cut_mean"]),
            ("cut in last 10%", (hrc >= 0.9).mean(), st["train"]["all"]["near_end_10"], st["val"]["all"]["near_end_10"]),
            ("cut in last 20%", (hrc >= 0.8).mean(), st["train"]["all"]["near_end_20"], st["val"]["all"]["near_end_20"])]
    for src in ["supreme_court", "high_courts", "tribunals", "acts"]:
        rows.append((f"share {src}", hmix[src] / len(ho), st["train"][src]["pairs"] / st["train"]["all"]["pairs"],
                     st["val"][src]["pairs"] / st["val"]["all"]["pairs"]))
    print("ok 4 distribution (held-out 990 | SFT train | SFT val):")
    for name, h, tr, va in rows:
        print(f"     {name:22s} {h:9.3f} | {tr:9.3f} | {va:9.3f}")
    h_mid, tr_mid = rows[3][1], rows[3][2]
    assert abs(h_mid - tr_mid) < 0.02 and abs(rows[5][1] - rows[5][2]) < 0.02 and abs(rows[0][1] - rows[0][2]) / rows[0][1] < 0.05
    assert all(abs(h - tr) < 0.01 for name, h, tr, _ in rows[7:])


def tiny_model():
    from models import build_model
    cfg = json.load(open(P("phase1", "configs", "smoke_archE.json")))["model"]
    cfg.update(d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16, ffn_hidden=128)
    torch.manual_seed(0)
    m, _ = build_model(cfg)
    m.doc_attn_impl = "dense"
    return m.eval()


def test_5_loss():
    recs = torch.load(os.path.join(PAIRS, "val.pt"), weights_only=False)
    recs = [r for r in recs if r["unsup"].any() and r["sup"].any()][:3]
    model = tiny_model()
    cfg = {"sft": {"mask_mode": "unsupported", "replay_ratio": 0.0}, "train": {"row_len": 3072, "rows_per_step": 2},
           "data": {"replay_mix": {"supreme_court": 1}}}

    def batch(mode, sw):
        rows = [[("p", 0), ("p", 1)], [("p", 2)]]
        plan = Plan.__new__(Plan)
        plan.row_len, plan.replay, plan.rsrc = 3072, {}, []
        return plan.materialize(rows, recs, mode, sw)

    out = {}
    for mode, sw in [("none", 2.0), ("unsupported", 2.0), ("unsupported+upweight", 1.0), ("unsupported+upweight", 2.0)]:
        x, y, w, rep = batch(mode, sw)
        with torch.no_grad():
            logits, _ = model(x)
            n = int((y != -100).sum())
            loss, nll = step_loss(logits, y, w, n)
            # manual: per example, nll of the unmasked target tokens, from the logits at the right positions
            man_num, man_den = 0.0, 0
            logp = torch.log_softmax(logits.float(), -1)
            for row_i, row in enumerate([[0, 1], [2]]):
                pos = 0
                for i in row:
                    r = recs[i]
                    Pn, T = len(r["prompt_ids"]), len(r["target_ids"])
                    for k in range(T):
                        masked = mode != "none" and r["unsup"][k]
                        if masked:
                            continue
                        wt = sw if (mode == "unsupported+upweight" and r["sup"][k]) else 1.0
                        man_num += -wt * logp[row_i, pos + Pn - 1 + k, int(r["target_ids"][k])].item()
                        man_den += 1
                    pos += Pn + T + 1
        assert man_den == n, (man_den, n)
        assert abs(loss.item() - man_num / man_den) < 1e-4, (mode, loss.item(), man_num / man_den)
        out[(mode, sw)] = loss.item()
    assert out[("unsupported+upweight", 1.0)] == out[("unsupported", 2.0)]
    assert out[("unsupported+upweight", 2.0)] != out[("unsupported", 2.0)]
    print(f"ok 5 loss: equals the manual mean over unmasked target tokens for every mode; "
          f"none {out[('none', 2.0)]:.5f}, unsupported {out[('unsupported', 2.0)]:.5f}, "
          f"upweight(w=1) {out[('unsupported+upweight', 1.0)]:.5f} (identical), upweight(w=2) {out[('unsupported+upweight', 2.0)]:.5f}")


def test_6_replay_ratio():
    recs = torch.load(os.path.join(PAIRS, "train.pt"), weights_only=False)[:8000]
    replay = load_replay(P("data", "sft", "replay_v1"))
    for basis, ratio in [("loss_tokens", 0.3), ("input_tokens", 0.3), ("loss_tokens", 0.1)]:
        cfg = {"sft": {"mask_mode": "unsupported", "supported_weight": 2.0, "replay_ratio": ratio, "replay_basis": basis},
               "train": {"row_len": 3072, "rows_per_step": 16, "epochs": 1},
               "data": {"replay_mix": {"supreme_court": 0.3, "high_courts": 0.4, "tribunals": 0.2, "acts": 0.1}}}
        plan = Plan(recs, replay, cfg, seed=0)
        if basis == "loss_tokens" and ratio == 0.3:     # the plan must not depend on mask_mode
            for mm in ["none", "unsupported+upweight"]:
                assert Plan(recs, replay, {**cfg, "sft": {**cfg["sft"], "mask_mode": mm}}, seed=0).steps == plan.steps
        sft = rep = sft_in = rep_in = 0
        src = Counter()
        for rows in plan.steps[:100]:
            x, y, w, r = plan.materialize(rows, recs, "none", 2.0)      # SFT side counted before masking
            valid = y != -100
            sft += int((valid & ~r).sum()); rep += int((valid & r).sum())
            rep_in += int(r.sum()); sft_in += int(((x != 0) & ~r).sum())
            for row in rows:
                for seg in row:
                    if seg[0] == "r":
                        src[plan.rsrc[seg[1]]] += seg[3]
        share = rep / (rep + sft) if basis == "loss_tokens" else rep_in / (rep_in + sft_in)
        assert abs(share - ratio) < 0.01, (basis, ratio, share)
        tot = sum(src.values())
        print(f"ok 6 replay ({basis}, ratio {ratio}): measured {share:.4f} over 100 batches "
              f"(replay source mix {', '.join(f'{k[:5]} {v / tot:.2f}' for k, v in src.items())}); "
              f"plan: {plan.stats['steps']} steps, pad share {plan.stats['pad_share']:.3f}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    for k, f in sorted((k, f) for k, f in globals().items() if k.startswith("test_")):
        if k.split("_")[1] in which:
            f()
    print("all requested tests passed")
