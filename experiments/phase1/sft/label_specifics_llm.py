"""Label the specifics of every SFT target with Qwen2.5-7B-Instruct (vLLM, bf16, temperature 0, guided JSON).

  conda env cs781-vllm (vLLM 0.8.5, torch 2.6/cu124; the cs781 env is left untouched)
  CUDA_VISIBLE_DEVICES=0 python phase1/sft/label_specifics_llm.py --split train --pilot 200      # pilot
  CUDA_VISIBLE_DEVICES=0 python phase1/sft/label_specifics_llm.py --split train                  # all (resumes)

Input: the masked-SFT pairs (data/sft/masked_sft_v2/{train,val}.pt; same builder, held-out exclusion and filters).
Output: data/sft/llm_labels_v1/<split>.jsonl, one record per pair (pair_id = split|source|doc_idx|part|cut_word), with
the raw model answer, the validated items (llm_labels.validate_items) and per-pair validation counts. Pairs already in
the file are skipped, so an interrupted run restarts where it stopped. Every pair passes llm_labels.HeldoutGuard before
it is sent (never a held-out / official-test prompt). Throughput -> <out>/label_log.jsonl.
--pilot N labels N pairs drawn uniformly (seed --pilot-seed) from the split; a later full run skips them.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import llm_labels as L  # noqa: E402

P = lambda *a: os.path.join(PROJECT_DIR, *a)
MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_MODEL_LEN = 8192


def load_pairs(pairs_dir, split):
    import torch
    recs = torch.load(os.path.join(pairs_dir, f"{split}.pt"), weights_only=False)
    return [(L.pair_id(r), {"idx": i, "source": r["source"], "prompt": r["prompt"], "target": r["target"]})
            for i, r in enumerate(recs)]


def select(pairs, a):
    if a.pilot:
        idx = np.sort(np.random.default_rng(a.pilot_seed).choice(len(pairs), size=a.pilot, replace=False))
        return [pairs[i] for i in idx]
    sel = pairs[a.start:]
    return sel[:a.limit] if a.limit else sel


class Labeler:
    def __init__(self, a, guard):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        self.guard, self.a = guard, a
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=MAX_MODEL_LEN, seed=0,
                       gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True,
                       guided_decoding_backend="xgrammar:disable-any-whitespace", max_num_seqs=a.max_num_seqs)
        self.SamplingParams, self.gd = SamplingParams, GuidedDecodingParams(json=L.SCHEMA)

    def label(self, chunk):
        """chunk: [(pid, rec)] -> list of output records (same order). A pair the guard refuses (e.g. it quotes the same
        30-word statute passage as a held-out / official test prompt) is never sent to the model: its record has
        skipped_guard=True and items=None (mask building then treats it as unlabeled)."""
        prompts, params, meta = [], [], []
        skipped = {}
        for pid, r in chunk:
            try:
                self.guard.check(pid, r["prompt"], r["target"])   # never a held-out / official test prompt
            except AssertionError as e:
                skipped[pid] = str(e)
                continue
            shown, trunc = L.truncate_prompt(r["prompt"], self.tok)
            text = self.tok.apply_chat_template(L.messages(shown, r["target"]), tokenize=False, add_generation_prompt=True)
            n_in = len(self.tok(text, add_special_tokens=False)["input_ids"])
            prompts.append(text)
            params.append(self.SamplingParams(temperature=0.0, max_tokens=max(256, min(self.a.max_new, MAX_MODEL_LEN - n_in)),
                                              guided_decoding=self.gd, seed=0))
            meta.append((shown, trunc, n_in))
        outs = iter(self.llm.generate(prompts, params, use_tqdm=False) if prompts else [])
        meta = iter(meta)
        recs = []
        for pid, r in chunk:
            if pid in skipped:
                recs.append(dict(pair_id=pid, idx=r["idx"], source=r["source"], skipped_guard=True, reason=skipped[pid],
                                 parse_ok=True, items=None, dropped=[], counts={}, qwen_in_tokens=0, qwen_out_tokens=0,
                                 finish_reason="skipped"))
                continue
            (shown, trunc, n_in), o = next(meta), next(outs)
            g = o.outputs[0]
            raw_items = L.parse_answer(g.text)
            salvaged = raw_items is None and g.finish_reason == "length"
            if salvaged:
                raw_items = L.parse_answer(g.text, salvage=True)
            items, counts, dropped = L.validate_items(raw_items or [], shown, r["target"])
            recs.append(dict(pair_id=pid, idx=r["idx"], source=r["source"], model=MODEL, qwen_in_tokens=n_in,
                             qwen_out_tokens=len(g.token_ids), prompt_truncated=trunc, finish_reason=g.finish_reason,
                             parse_ok=raw_items is not None, salvaged=salvaged, raw=g.text, items=items,
                             dropped=dropped, counts=counts))
        return recs


def run_labeling(work, label_fn, path, chunk_size, log_path=None, split="", claim=None):
    """Label `work` ([(pid, rec)], the whole selection) chunk by chunk with label_fn, appending one JSONL record per
    pair to `path` (flushed + fsynced per chunk, so an interruption loses at most the current chunk).
    claim(k) -> bool: take chunk k only if this process claims it (several processes share one selection; chunk k is
    the same pairs in every process because chunks are cut from the selection, not from what is left to do).
    done is re-read from all stores before each chunk, so pairs labeled by an earlier launch are skipped.
    -> (n_done, start time)."""
    logf = open(log_path, "a") if log_path else None
    n_done, t_start = 0, time.time()
    labels_dir = os.path.dirname(path)
    with open(path, "a") as f:
        for k, i in enumerate(range(0, len(work), chunk_size)):
            if claim is not None and not claim(k):
                continue
            done = L.load_done_dir(labels_dir, split) if split else L.load_done(path)
            chunk = [(pid, r) for pid, r in work[i:i + chunk_size] if pid not in done]
            if not chunk:
                continue
            tc = time.time()
            recs = label_fn(chunk)
            assert [r["pair_id"] for r in recs] == [pid for pid, _ in chunk]
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            n_done += len(recs)
            el = time.time() - t_start
            rate = n_done / el * 60
            log = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), split=split, file=os.path.basename(path), chunk=k,
                       chunk_pairs=len(recs), chunk_s=time.time() - tc, done=n_done, pairs_per_min=rate,
                       parse_fail=sum(not r.get("parse_ok", True) for r in recs),
                       truncated_answers=sum(r.get("finish_reason") == "length" for r in recs),
                       salvaged=sum(bool(r.get("salvaged")) and r.get("parse_ok", True) for r in recs),
                       out_tokens_mean=float(np.mean([r.get("qwen_out_tokens", 0) for r in recs])),
                       in_tokens_mean=float(np.mean([r.get("qwen_in_tokens", 0) for r in recs])))
            if logf:
                logf.write(json.dumps(log) + "\n"); logf.flush()
            print(json.dumps(log), flush=True)
    return n_done, t_start


def claimer(claim_dir):
    """claim(k): atomically create <claim_dir>/<k> (O_EXCL); True for exactly one process per k."""
    os.makedirs(claim_dir, exist_ok=True)

    def claim(k):
        try:
            os.close(os.open(os.path.join(claim_dir, str(k)), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return True
        except FileExistsError:
            return False
    return claim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=P("data", "sft", "masked_sft_v2"))
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--out", default=P("data", "sft", "llm_labels_v1"))
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--pilot-seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=1024, help="pairs per generate call (one JSONL flush per chunk)")
    ap.add_argument("--max-new", type=int, default=1536)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--tag", default="", help="write <split>.<tag>.jsonl (one file per process)")
    ap.add_argument("--claim-dir", default="", help="share the selection with other processes using this claim dir "
                    "(use a NEW dir per launch; already-labeled pairs are skipped anyway)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.split}.{a.tag}.jsonl" if a.tag else f"{a.split}.jsonl")
    t0 = time.time()
    pairs = load_pairs(a.pairs, a.split)
    other = load_pairs(a.pairs, "val" if a.split == "train" else "train")
    allowed = [pid for pid, _ in pairs] + [pid for pid, _ in other]
    del other
    guard = L.load_guard(PROJECT_DIR, allowed)
    L.repair_tail(path)
    done = L.load_done_dir(a.out, a.split)
    work = select(pairs, a)
    left = sum(pid not in done for pid, _ in work)
    print(f"{len(pairs)} {a.split} pairs, selected {len(work)}, already labeled (all stores) {len(done)}, to do {left} "
          f"({time.time() - t0:.0f}s)", flush=True)
    if not left:
        return
    lab = Labeler(a, guard)
    n_done, t_start = run_labeling(work, lab.label, path, a.chunk, os.path.join(a.out, "label_log.jsonl"), a.split,
                                   claim=claimer(a.claim_dir) if a.claim_dir else None)
    print(f"done: {n_done} pairs in {(time.time() - t_start) / 60:.1f} min, guard checked {guard.checked}", flush=True)


if __name__ == "__main__":
    main()
