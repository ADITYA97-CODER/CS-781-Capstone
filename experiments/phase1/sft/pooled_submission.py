"""Pooled MBR over 16 candidates per prompt: 8 from Arch E + 8 from the masked-SFT upweight model, both decoded with
T 0.7 / top-p 0.9 / rep 1.1 + header & citation steering (seeds 1-8), selected with phase1/mbr/mbr_rerank.py and the
frozen penalty weights (outputs/phase1/mbr_rerank/archE_local_rerank8_tuning.json).

  python phase1/sft/pooled_submission.py --stage local      # rerank the existing local samples -> local score vs Arch E x8
  python phase1/sft/pooled_submission.py --stage official --gpus 0,1   # generate upweight official seeds 1-8, rerank, NEW submission

Arch E official samples: outputs/phase1/archE_steer_official/s1-s8 (same flags). Output (never overwritten):
outputs/phase1/masked_sft/official/pooled16/{mbr_rerank16.json, prediction.json, submission.zip}
"""
import argparse
import json
import os
import subprocess
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_masked_sft import MODELS, OUT, P, PY, SEEDS, WEIGHTS, gen_jobs, run_jobs  # noqa: E402

OFF = P("outputs", "phase1", "masked_sft", "official")
ARCHE_OFFICIAL = P("outputs", "phase1", "archE_steer_official")


def rerank16(cands, prompts, out, gpu):
    if os.path.exists(out):
        return out
    w = json.load(open(WEIGHTS))["weights"]
    subprocess.run([PY, P("phase1/mbr/mbr_rerank.py"), "--cands", *cands, "--prompts", prompts, "--weights", json.dumps(w),
                    "--out", out, "--batch-size", "64"], check=True, cwd=P(), env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
    return out


def stage_local(a):
    cands = [os.path.join(OUT, m, f"s{s}", "prediction.json") for m in ("start", "sft_upweight") for s in SEEDS]
    out = rerank16(cands, "local", os.path.join(OUT, "pooled16_mbr16.json"), a.gpus.split(",")[-1])
    # local score vs Arch E x8 (same metric code as the report)
    sys.path.insert(0, P("phase1", "diagnostics"))
    from eval_fact_decoding import Ctx, compute_metrics, paired_diff_ci
    ctx = Ctx()
    ids = list(ctx.refs)
    res = {}
    for name, path, dirs in [("start_x8", os.path.join(OUT, "start_mbr8.json"), [os.path.join(OUT, "start", f"s{s}") for s in SEEDS]),
                             ("upweight_x8", os.path.join(OUT, "sft_upweight_mbr8.json"), [os.path.join(OUT, "sft_upweight", f"s{s}") for s in SEEDS]),
                             ("pooled16", out, [os.path.dirname(c) for c in cands])]:
        res[name] = compute_metrics(ctx, path, ids, dirs)
    pred = json.load(open(out))
    up = set()
    for s in SEEDS:   # how often the pooled pick came from the upweight model
        d = json.load(open(os.path.join(OUT, "sft_upweight", f"s{s}", "prediction.json")))
        up |= {k for k in ids if d[k]["text"] == pred[k]["text"]}
    lines = ["| local, 990 prompts | BERTScore | Δ vs Arch E x8 (95% CI) | ROUGE-L | invented rate | invented/output | reuse | closing | loops | copy8 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        d, ci = paired_diff_ci(r["bertscore_per_prompt"], res["start_x8"]["bertscore_per_prompt"])
        dd = "—" if name == "start_x8" else f"{d:+.4f} ({ci[0]:+.4f}…{ci[1]:+.4f})"
        lines.append(f"| {name} | {r['bertscore']:.4f} | {dd} | {r['rouge_l']:.4f} | {r['invented_rate']:.3f} | "
                     f"{r['invented_per_output']:.2f} | {r['reuse_overall']:.3f} | {r['closing']} | {r['loops']} | {r['copy8']:.4f} |")
    lines.append(f"\nPooled picks taken from the upweight model's samples: {len(up)}/{len(ids)}")
    open(os.path.join(OUT, "REPORT_pooled16.md"), "w").write("# Pooled MBR x16 (Arch E 8 + SFT upweight 8), local\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


def stage_official(a):
    od = os.path.join(OFF, "pooled16")
    if os.path.exists(os.path.join(od, "prediction.json")):
        sys.exit(f"{od}/prediction.json exists; not overwriting")
    os.makedirs(od, exist_ok=True)
    gpus = [int(g) for g in a.gpus.split(",")]
    run_jobs(gen_jobs("sft_upweight", MODELS["sft_upweight"], "official", OFF), gpus)
    cands = ([os.path.join(ARCHE_OFFICIAL, f"s{s}", "prediction.json") for s in SEEDS]
             + [os.path.join(OFF, "sft_upweight", f"s{s}", "prediction.json") for s in SEEDS])
    for c in cands:
        assert os.path.exists(c), c
    mbr = rerank16(cands, "official", os.path.join(od, "mbr_rerank16.json"), gpus[0])
    test = json.load(open(P("data", "Phase_1_Pretraining", "Test", "test.json")))
    pred = json.load(open(mbr))
    short = [k for k in test if len(pred[k]["text"].split()) < 128]
    if short:
        sys.exit(f"outputs under 128 words for ids {short}: regenerate them with --max-new-tokens 480 before submitting")
    pred = {k: {"text": " ".join(pred[k]["text"].split()[:128])} for k in test}
    n = [len(v["text"].split()) for v in pred.values()]
    assert len(pred) == 1000 and list(pred) == list(test) and min(n) == max(n) == 128
    json.dump(pred, open(os.path.join(od, "prediction.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    with zipfile.ZipFile(os.path.join(od, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(od, "prediction.json"), arcname="prediction.json")
    z = zipfile.ZipFile(os.path.join(od, "submission.zip"))
    assert z.namelist() == ["prediction.json"] and z.testzip() is None and json.loads(z.read("prediction.json")) == pred
    up = set()
    for s in SEEDS:
        d = json.load(open(os.path.join(OFF, "sft_upweight", f"s{s}", "prediction.json")))
        up |= {k for k in test if " ".join(d[k]["text"].split()[:128]) == pred[k]["text"]}
    print(f"official submission: {od}/submission.zip | 1000 ids in test order, all exactly 128 words, zip verified | "
          f"picks from the upweight model: {len(up)}/1000")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["local", "official"])
    ap.add_argument("--gpus", default="0,1")
    a = ap.parse_args()
    {"local": stage_local, "official": stage_official}[a.stage](a)


if __name__ == "__main__":
    main()
