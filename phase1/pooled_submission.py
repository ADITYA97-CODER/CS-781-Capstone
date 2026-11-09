"""Phase-1 inference: pooled minimum-Bayes-risk selection over 16 sampled continuations per official prompt.

8 samples come from Arch E and 8 from the masked-SFT (upweight) model, both decoded with T 0.7 / top-p 0.9 /
repetition penalty 1.1 plus header and citation steering. The penalty-aware reranker picks one candidate per prompt.

  python phase1/pooled_submission.py --gpu 0        # -> outputs/phase1/pooled16/{prediction.json, submission.zip}
"""
import argparse
import json
import os
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "phase1", "pooled16")
TEST = os.path.join(ROOT, "data", "Phase_1_Pretraining", "Test", "test.json")
MODELS = {"arch_e": "runs/archE_diff_145M/ckpt_best.pt", "sft_upweight": "runs/sft_archE_v2_upweight/ckpt_best.pt"}
SEEDS = range(1, 9)
RERANK_WEIGHTS = {"header": 0, "close": 0.01, "sign": 0.03, "invent": 0.001}     # tuned on local prompts, frozen
GEN = ["--temperature", "0.7", "--top-p", "0.9", "--repetition-penalty", "1.1", "--steer-headers", "--steer-cite"]


def run(cmd, gpu):
    subprocess.run([sys.executable, *cmd], check=True, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})


def generate(gpu, ckpt, seed, out, extra):
    run(["phase1/generate.py", "--ckpt", ckpt, "--prompts", "official", *GEN, "--seed", str(seed), "--out", out, *extra], gpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    test = json.load(open(TEST))
    cands = []
    for name, ckpt in MODELS.items():
        for s in SEEDS:
            d = os.path.join(OUT, name, f"s{s}")
            if not os.path.exists(os.path.join(d, "prediction.json")):
                generate(a.gpu, ckpt, s, d, ["--batch-size", "128", "--max-new-tokens", "320", "--max-mem-gb", "26"])
            cands.append(os.path.join(d, "prediction.json"))

    # A prompt whose 16 candidates are all under 128 words (number-heavy tables) is regenerated with a larger token budget.
    preds = [json.load(open(c)) for c in cands]
    short = [k for k in test if all(len(p[k]["text"].split()) < 128 for p in preds)]
    if short:
        print("regenerating prompts with no 128-word candidate:", short, flush=True)
        subset = os.path.join(OUT, "short_ids.json")
        json.dump(short, open(subset, "w"))
        for c, p, (name, s) in zip(cands, preds, [(n, s) for n in MODELS for s in SEEDS]):
            d = os.path.join(OUT, name, f"s{s}_long")
            generate(a.gpu, MODELS[name], s, d, ["--batch-size", "1", "--max-new-tokens", "480", "--subset", subset])
            new = json.load(open(os.path.join(d, "prediction.json")))
            p.update({k: new[k] for k in short})
            json.dump(p, open(c, "w"), ensure_ascii=False, indent=2)

    mbr = os.path.join(OUT, "mbr_rerank16.json")
    run(["phase1/mbr/mbr_rerank.py", "--cands", *cands, "--prompts", "official", "--weights", json.dumps(RERANK_WEIGHTS),
         "--out", mbr, "--batch-size", "64"], a.gpu)
    picked = json.load(open(mbr))
    bad = [k for k in test if len(picked[k]["text"].split()) < 128]
    assert not bad, f"outputs under 128 words: {bad}"
    pred = {k: {"text": " ".join(picked[k]["text"].split()[:128])} for k in test}
    path = os.path.join(OUT, "prediction.json")
    json.dump(pred, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    with zipfile.ZipFile(os.path.join(OUT, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, arcname="prediction.json")
    print(f"wrote {len(pred)} ids, all exactly 128 words -> {OUT}/submission.zip")


if __name__ == "__main__":
    main()
