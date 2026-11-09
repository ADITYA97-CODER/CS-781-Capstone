"""Tests for the LLM specific labels and mask_source masks (phase1/sft/llm_labels.py, llm_masks.py,
label_specifics_llm.py).   python phase1/tests/test_llm_labels.py   (CPU only, env cs781)

1 span mapping   LLM texts -> char spans -> legal_bpe_32k token spans: every sub-word of every occurrence is masked,
                 nothing else; "Ram" never matches inside "Ramesh"
2 union          spans the LLM marks supported WITH verified evidence are never masked in "union" (also when the
                 regex calls them unsupported); llm / union / regex flags as specified
3 validation     non-verbatim items dropped, supported items with missing / non-verbatim evidence relabeled
                 unsupported, stoplist phrases dropped, duplicates merged, truncated answers salvaged
4 guard          no Qwen call is ever made for a held-out or official test prompt (fake LLM records every call; a
                 non-SFT pair id is refused); every labeled pair on disk is an SFT pair with no 30-word overlap
5 resume         interrupting labeling (exception mid-run + a torn last line) and restarting labels every pair
                 exactly once
6 train flags    mask_source "regex" leaves the train labels byte-identical; apply_masks round-trips the flags
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_DIR, "phase1"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "phase1", "sft"))
import llm_labels as L  # noqa: E402
import llm_masks as M  # noqa: E402
from label_specifics_llm import Labeler, run_labeling  # noqa: E402
from train_sft import pair_segment  # noqa: E402

P = lambda *a: os.path.join(PROJECT_DIR, *a)
PAIRS = P("data", "sft", "masked_sft_v2")
LABELS = P("data", "sft", "llm_labels_v1")


def item(text, target, label="unsupported", evidence=None, evidence_ok=False, typ="name"):
    spans, _ = L.find_spans(text, target)
    return dict(text=text, type=typ, label=label, evidence=evidence, evidence_ok=evidence_ok, spans=spans)


def masked_text(target, offs, flag):
    return [target[a:b] for (a, b), f in zip(offs, flag) if f]


def test_1_span_mapping():
    target = ("The witness Ram Lal stated that Ramesh and Ram Lal met on 12.03.1998 near Bulandshahar; "
              "Ram Lal's statement (Ex.P-12) was recorded under Section 302/34 IPC.")
    ids, offs = M.target_offsets(target)
    its = [item("Ram Lal", target), item("Ex.P-12", target, typ="exhibit"), item("Bulandshahar", target, typ="place")]
    assert len(its[0]["spans"]) == 3, its[0]["spans"]                      # all three occurrences
    uns, sup = M.flags(offs, its, "llm", np.zeros(len(ids), bool), np.zeros(len(ids), bool))
    assert not sup.any()
    for it in its:
        for a, b in it["spans"]:
            cover = [(s, e) for s, e in offs if s < b and e > a]
            assert cover and all(uns[i] for i, (s, e) in enumerate(offs) if s < b and e > a), it["text"]
            # the union of the masked tokens' characters covers the whole span (every sub-word)
            got = set().union(*[set(range(s, e)) for s, e in cover])
            assert set(range(a, b)) <= got, (it["text"], target[a:b])
    # nothing outside the spans is masked (tokens may include a leading space only)
    for i, (s, e) in enumerate(offs):
        inside = any(s < b and e > a for it in its for a, b in it["spans"])
        assert uns[i] == inside, (i, target[s:e])
    # "Ram" must not match inside "Ramesh"; whole-word occurrences only
    spans, anywhere = L.find_spans("Rames", target)
    assert spans == [] and anywhere
    assert len(L.find_spans("Ram", target)[0]) == 3
    words = masked_text(target, offs, uns)
    print(f"ok 1 span mapping: {len(ids)} target tokens, {int(uns.sum())} masked -> {''.join(words)!r}")


def test_2_union_keeps_verified_supported():
    target = "The appellant Ram Lal was convicted under Section 302 IPC on 12.03.1998 by Judge Mohan Singh."
    ids, offs = M.target_offsets(target)
    n = len(ids)
    regex_uns = M.span_hits(offs, L.find_spans("Ram Lal", target)[0] + L.find_spans("12.03.1998", target)[0])
    regex_sup = M.span_hits(offs, L.find_spans("302", target)[0])
    its = [item("Ram Lal", target, "supported", "Ramlal", True),           # regex says unsupported, LLM verified
           item("Mohan Singh", target, "unsupported"),
           item("302", target, "inferable"),
           item("12.03.1998", target, "supported", "12 March", False)]     # unverified evidence: no protection
    rl = M.span_hits(offs, its[0]["spans"])
    for src in ("llm", "union", "regex"):
        uns, sup = M.flags(offs, its, src, regex_uns, regex_sup)
        assert not (uns & sup).any()
        if src == "union":
            assert not (uns & rl).any(), "verified-supported span masked in union"
            assert uns[M.span_hits(offs, its[1]["spans"])].all() and uns[M.span_hits(offs, its[3]["spans"])].all()
            assert sup[rl].all()
        if src == "llm":
            assert uns[M.span_hits(offs, its[1]["spans"])].all() and not uns[M.span_hits(offs, its[2]["spans"])].any()
        if src == "regex":
            assert (uns == regex_uns).all() and (sup == regex_sup).all()
    # randomized: no verified-supported token is ever masked in union
    rng = np.random.default_rng(0)
    for _ in range(300):
        words = target.split()
        texts = {" ".join(words[i:i + rng.integers(1, 3)]) for i in rng.integers(0, len(words), 4)}
        its = [item(t, target, str(rng.choice(L.LABELS)), "x", bool(rng.random() < 0.7)) for t in texts]
        ru = rng.random(n) < 0.3
        uns, _ = M.flags(offs, its, "union", ru, np.zeros(n, bool))
        keep = M.span_hits(offs, [sp for it in its if it["label"] == "supported" and it["evidence_ok"] for sp in it["spans"]])
        assert not (uns & keep).any()
    print("ok 2 union: LLM-supported spans with verified evidence are never masked (fixed case + 300 random cases)")


def test_3_validation():
    prompt = "The appellant Ram Lal was convicted by the Sessions Judge, Jaipur under Section 302 read with Section 34 IPC."
    target = "Ramlal was sentenced on 12.03.1998 under 302/34 IPC by the Court. Ramlal appealed. Ramesh too."
    raw = [
        {"text": "Ramlal", "type": "name", "label": "supported", "evidence": "Ram Lal"},          # kept, verified
        {"text": "302/34 IPC", "type": "section", "label": "supported", "evidence": "Section 302 read with Section 34"},
        {"text": "12.03.1998", "type": "date", "label": "supported", "evidence": "on 12.03.1998"},  # evidence from TARGET
        {"text": "Jaipur", "type": "place", "label": "supported", "evidence": "Jaipur"},          # not in target
        {"text": "Ram Lal", "type": "name", "label": "unsupported", "evidence": None},            # not verbatim in target
        {"text": "Rames", "type": "name", "label": "unsupported", "evidence": None},              # only inside a word
        {"text": "the Court", "type": "court", "label": "inferable", "evidence": None},           # stoplist
        {"text": "Ramlal", "type": "name", "label": "unsupported", "evidence": None},             # duplicate
        {"text": "Ramesh", "type": "name", "label": "supported", "evidence": None},               # no evidence
        {"text": "x", "type": "bogus", "label": "supported", "evidence": None},                   # bad type
    ]
    items, c, dropped = L.validate_items(raw, prompt, target)
    by = {it["text"]: it for it in items}
    assert set(by) == {"Ramlal", "302/34 IPC", "12.03.1998", "Ramesh"}, set(by)
    assert by["Ramlal"]["label"] == "supported" and by["Ramlal"]["evidence_ok"] and len(by["Ramlal"]["spans"]) == 2
    assert by["302/34 IPC"]["label"] == "supported"
    assert by["12.03.1998"]["label"] == "unsupported" and by["12.03.1998"]["orig_label"] == "supported"
    assert by["Ramesh"]["label"] == "unsupported"
    assert c["not_verbatim"] == 2 and c["no_word_match"] == 1 and c["stoplist"] == 1 and c["evidence_fail"] == 2
    assert c["duplicate"] == 1 and c["label_conflict"] == 1 and c["bad_item"] == 1 and c["kept"] == 4
    assert c["evidence_in_target_only"] == 1
    assert {d["why"] for d in dropped} == {"not_verbatim", "no_word_match", "stoplist"}
    # salvage: an answer cut off in a loop keeps its complete items
    cut = '{"specifics": [{"text": "A", "type": "name", "evidence": null, "label": "unsupported"}, {"text": "B", "ty'
    assert L.parse_answer(cut) is None and [x["text"] for x in L.parse_answer(cut, salvage=True)] == ["A"]
    assert L.parse_answer('{"specifics": []}') == [] and L.parse_answer("nonsense", salvage=True) is None
    print(f"ok 3 validation: counts {dict((k, v) for k, v in c.items() if v)}")


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def generate(self, prompts, params, use_tqdm=False):
        from types import SimpleNamespace
        self.prompts += prompts
        g = SimpleNamespace(text='{"specifics": []}', token_ids=[1, 2, 3], finish_reason="stop")
        return [SimpleNamespace(outputs=[g]) for _ in prompts]


def fake_labeler(guard):
    from transformers import AutoTokenizer
    lab = object.__new__(Labeler)
    lab.guard, lab.a = guard, type("A", (), {"max_new": 64})()
    lab.tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    lab.llm, lab.SamplingParams, lab.gd = FakeLLM(), (lambda **k: k), None
    return lab


def test_4_guard():
    ho = json.load(open(P("data", "corpus", "eval", "local_test.json")))
    test = json.load(open(P("data", "Phase_1_Pretraining", "Test", "test.json")))
    val = torch.load(os.path.join(PAIRS, "val.pt"), weights_only=False)
    ok_pairs = [(L.pair_id(r), {"idx": i, "source": r["source"], "prompt": r["prompt"], "target": r["target"]})
                for i, r in enumerate(val[:5])]
    guard = L.load_guard(PROJECT_DIR, [pid for pid, _ in ok_pairs] + ["train|high_courts|1|0|600"])
    lab = fake_labeler(guard)
    recs = lab.label(ok_pairs)
    assert len(recs) == 5 and len(lab.llm.prompts) == 5
    refused = 0
    for name, d in [("held-out", ho), ("official test", test)]:
        for k in list(d)[:: max(1, len(d) // 50)]:
            text = d[k]["text"]
            # a held-out / test prompt can never get through: neither under an allowed pair id nor an unknown one
            for pid in ["train|high_courts|1|0|600", f"heldout|{name}|{k}|0|0"]:
                n_before = len(lab.llm.prompts)
                out = lab.label([(pid, {"idx": 0, "source": "x", "prompt": text, "target": "abc"})] + ok_pairs[:1])
                assert out[0]["skipped_guard"] and out[0]["items"] is None, f"guard let a {name} prompt through"
                assert not out[1].get("skipped_guard")
                assert len(lab.llm.prompts) == n_before + 1, "LLM called for a forbidden prompt"
                assert text not in "".join(lab.llm.prompts[n_before:])
                refused += 1
    # an allowed id whose prompt merely contains a held-out passage is refused too
    k = next(iter(ho))
    sneaky = "intro words " + " ".join(ho[k]["text"].split()[100:140]) + " outro"
    try:
        guard.check("train|high_courts|1|0|600", sneaky)
        raise RuntimeError("guard missed an embedded held-out passage")
    except AssertionError:
        refused += 1
    # every labeled pair on disk: an SFT pair id, no overlap with held-out / official test prompts
    n_disk = n_skip = 0
    for split in ("train", "val"):
        if not L.split_files(LABELS, split):
            continue
        pairs = {L.pair_id(r): r for r in torch.load(os.path.join(PAIRS, f"{split}.pt"), weights_only=False)}
        g2 = L.load_guard(PROJECT_DIR, pairs)
        for pid, rec in L.load_records_dir(LABELS, split).items():
            assert pid in pairs
            if rec.get("skipped_guard"):
                assert rec["items"] is None and rec["qwen_out_tokens"] == 0
                n_skip += 1
                continue
            g2.check(pid, pairs[pid]["prompt"], pairs[pid]["target"])
            n_disk += 1
    print(f"ok 4 guard: {refused} forbidden prompts refused with 0 LLM calls; 5 SFT pairs labeled; "
          f"{n_disk} labeled pairs on disk verified (SFT ids, no held-out/test overlap), {n_skip} guard-skipped records")


def test_5_resume():
    work = [(f"train|acts|{i}|0|600", {"idx": i}) for i in range(53)]
    calls = []

    def label_fn(chunk, fail_at=None):
        calls.extend(pid for pid, _ in chunk)
        if fail_at is not None and any(pid == fail_at for pid, _ in chunk):
            raise KeyboardInterrupt
        return [dict(pair_id=pid, items=[], parse_ok=True) for pid, _ in chunk]

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "train.jsonl")
        try:
            run_labeling(L.todo(work, L.load_done(path)), lambda c: label_fn(c, fail_at=work[23][0]), path, 10)
        except KeyboardInterrupt:
            pass
        with open(path, "a") as f:
            f.write('{"pair_id": "train|acts|20|0|600", "ite')                    # torn write
        L.repair_tail(path)
        done = L.load_done(path)
        assert done == {pid for pid, _ in work[:20]}, len(done)
        rest = L.todo(work, done)
        assert [pid for pid, _ in rest] == [pid for pid, _ in work[20:]]
        run_labeling(rest, label_fn, path, 10)
        ids = [json.loads(line)["pair_id"] for line in open(path)]
        assert len(ids) == len(set(ids)) == len(work) and set(ids) == {pid for pid, _ in work}
        assert L.todo(work, L.load_done(path)) == []
    # two processes (own files, one claim dir) share a selection; one is interrupted, both relaunched (new claim dir)
    from label_specifics_llm import claimer
    with tempfile.TemporaryDirectory() as d:
        pa, pb = os.path.join(d, "train.gpu0.jsonl"), os.path.join(d, "train.gpu1.jsonl")
        ca = claimer(os.path.join(d, "claims1"))
        try:
            run_labeling(work, lambda c: label_fn(c, fail_at=work[35][0]), pa, 10, split="train", claim=ca)
        except KeyboardInterrupt:
            pass
        run_labeling(work, label_fn, pb, 10, split="train", claim=ca)       # takes every chunk not claimed by A
        c2 = claimer(os.path.join(d, "claims2"))
        for p_ in (pa, pb):
            L.repair_tail(p_)
            run_labeling(work, label_fn, p_, 10, split="train", claim=c2)
        ids = [json.loads(line)["pair_id"] for p_ in (pa, pb) for line in open(p_)]
        assert len(ids) == len(set(ids)) == len(work), (len(ids), len(set(ids)))
        assert set(L.load_records_dir(d, "train")) == {pid for pid, _ in work}
    print(f"ok 5 resume: interrupted at pair 23 (chunk 3) + torn line -> restart labeled the remaining {len(rest)}; "
          f"{len(work)} pairs, each exactly once; 2 processes with chunk claims + interruption + relaunch: each once")


def test_6_train_flags():
    recs = torch.load(os.path.join(PAIRS, "val.pt"), weights_only=False)[:200]
    before = [pair_segment(r, "unsupported+upweight", 2.0) for r in recs]
    records = {}
    for r in recs:
        its = [item(w, r["target"], lab) for w, lab in zip(r["target"].split()[3:9:2], ["unsupported", "supported", "inferable"])]
        records[L.pair_id(r)] = {"items": [dict(it, evidence_ok=it["label"] == "supported") for it in its]}
    out, stats = M.build(recs, records, "regex")
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "m.pt")
        torch.save(out, f)
        copy = [dict(r) for r in recs]
        assert M.apply_masks(copy, f) == "regex"
        for r, b in zip(copy, before):
            a = pair_segment(r, "unsupported+upweight", 2.0)
            assert all(np.array_equal(x, y) for x, y in zip(a, b))
        out2, st2 = M.build(recs, records, "llm")
        torch.save(out2, f)
        copy = [dict(r) for r in recs]
        M.apply_masks(copy, f)
        for r in copy:
            ids, offs = M.target_offsets(r["target"])
            u, s = M.flags(offs, records[L.pair_id(r)]["items"], "llm", r["unsup"], r["sup"])
            assert np.array_equal(u, r["unsup"]) and np.array_equal(s, r["sup"])
    print(f"ok 6 train flags: regex masks byte-identical labels/weights on 200 pairs; llm masks round-trip "
          f"(llm masked {st2['all']['share_masked']:.3f} vs regex {st2['all']['regex_share_masked']:.3f} of tokens)")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    for k, f in sorted((k, f) for k, f in globals().items() if k.startswith("test_")):
        if k.split("_")[1] in which:
            f()
    print("all requested tests passed")
