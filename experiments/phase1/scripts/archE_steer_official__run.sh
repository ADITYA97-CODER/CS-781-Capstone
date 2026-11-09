#!/bin/bash
# Official submission: Arch E (runs/archE_diff_145M/ckpt_best.pt) + header & citation steering,
# 8 samples (seeds 1-8, T 0.7, top-p 0.9, rep 1.1, batch 128), penalty-aware MBR
# (phase1/mbr/mbr_rerank.py, weights tuned on even-index local prompts with Arch E MBR x8). GPU 0.
set -e
export CUDA_VISIBLE_DEVICES=0
cd $(git rev-parse --show-toplevel)
PY="python"
CKPT=runs/archE_diff_145M/ckpt_best.pt
O=outputs/phase1/archE_steer_official
W=$(python3 -c "import json;print(json.dumps(json.load(open('outputs/phase1/mbr_rerank/archE_local_rerank8_tuning.json'))['weights']))")
for s in 1 2 3 4 5 6 7 8; do
  if [ ! -f $O/s$s/prediction.json ]; then
    $PY phase1/generate.py --ckpt $CKPT --prompts official --temperature 0.7 --top-p 0.9 --repetition-penalty 1.1 \
        --batch-size 128 --max-new-tokens 320 --max-mem-gb 26 --steer-headers --steer-cite --seed $s --out $O/s$s > $O/s$s.log 2>&1
  fi
  echo "$(date +%T) seed $s done: $(grep -o '"seconds": [0-9.]*' $O/s$s/generation_stats.json) $(grep -o 'WARNING: [0-9]*' $O/s$s.log || echo 'all >=128 words')"
done
$PY phase1/mbr/mbr_rerank.py --cands $O/s{1,2,3,4,5,6,7,8}/prediction.json --prompts official --weights "$W" \
    --out $O/mbr_rerank8.json --batch-size 64 2>&1 | grep -E "wrote"
python3 - <<PYEOF
import json, zipfile
test = json.load(open("data/Phase_1_Pretraining/Test/test.json"))
pred = json.load(open("$O/mbr_rerank8.json"))
pred = {k: {"text": pred[k]["text"]} for k in test}
n = [len(v["text"].split()) for v in pred.values()]
assert len(pred) == 1000 and list(pred) == list(test) and min(n) == max(n) == 128
json.dump(pred, open("$O/prediction.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
with zipfile.ZipFile("$O/submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write("$O/prediction.json", arcname="prediction.json")
z = zipfile.ZipFile("$O/submission.zip")
assert z.namelist() == ["prediction.json"] and z.testzip() is None and json.loads(z.read("prediction.json")) == pred
print(f"official submission: {len(pred)} ids in test order, all exactly 128 words, zip verified")
PYEOF
echo "$(date) ALL_DONE"
