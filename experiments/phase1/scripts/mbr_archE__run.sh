#!/bin/bash
# Waits for the Arch E training run (PID 3628142) to finish, then:
#   8 sampled continuations (T 0.7, top-p 0.9, rep 1.1, seeds 1-8) on the local and official prompts,
#   local scores (single sample, MBR x4, MBR x8), and the official MBR x8 prediction.json + submission.zip.
# Same recipe as outputs/phase1/mbr_docmask/run.sh (Arch D doc-mask). GPU 0 only.
set -e
export CUDA_VISIBLE_DEVICES=0
cd $(git rev-parse --show-toplevel)
PY="python"
TRAIN_PID=3628142
TRAIN_LOG=runs/archE_diff_145M.log
CKPT=runs/archE_diff_145M/ckpt_best.pt
M=outputs/phase1/mbr_archE
EVAL() { echo "== $1"; $PY "official_eval/Evaluation Script/evaluate_phase1.py" --truth data/corpus/eval/local_ref.json --prediction $2 2>/dev/null | grep -E "BERT|ROUGE|BLEU"; }

echo "$(date) waiting for training PID $TRAIN_PID"
while kill -0 $TRAIN_PID 2>/dev/null; do sleep 60; done
if [ "$(tail -1 $TRAIN_LOG)" != "done" ]; then
  echo "$(date) TRAINING DID NOT FINISH CLEANLY (last log line: $(tail -1 $TRAIN_LOG)); not generating"; exit 1
fi
echo "$(date) training finished: $(grep 'new best val' $TRAIN_LOG | tail -1)"
sleep 30   # let the GPU memory free up

for set in local official; do
  for s in 1 2 3 4 5 6 7 8; do
    $PY phase1/generate.py --ckpt $CKPT --prompts $set --temperature 0.7 --top-p 0.9 --repetition-penalty 1.1 \
        --seed $s --batch-size 128 --out $M/${set}_s$s > $M/${set}_s$s.log 2>&1
    echo "$(date +%T) generated $set seed $s: $(grep WARNING $M/${set}_s$s.log || echo 'all >=128 words')"
  done
done
EVAL "local single sample seed 1" $M/local_s1/prediction.json
$PY phase1/mbr/mbr_select.py --cands $M/local_s{1,2,3,4}/prediction.json --out $M/local_mbr4.json 2>&1 | grep -E "picked|consensus"
EVAL "local MBR x4" $M/local_mbr4.json
$PY phase1/mbr/mbr_select.py --cands $M/local_s{1,2,3,4,5,6,7,8}/prediction.json --out $M/local_mbr8.json 2>&1 | grep -E "picked|consensus"
EVAL "local MBR x8" $M/local_mbr8.json
mkdir -p $M/official
$PY phase1/mbr/mbr_select.py --cands $M/official_s{1,2,3,4,5,6,7,8}/prediction.json --out $M/official/prediction.json 2>&1 | grep -E "picked|consensus"
python3 - <<EOF
import json, zipfile
with zipfile.ZipFile("$M/official/submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write("$M/official/prediction.json", arcname="prediction.json")
d = json.load(zipfile.ZipFile("$M/official/submission.zip").open("prediction.json"))
test = json.load(open("data/Phase_1_Pretraining/Test/test.json"))
n = [len(v["text"].split()) for v in d.values()]
print(f"official submission: {len(d)} ids, ids match test set: {set(d) == set(test)}, "
      f"words min/max {min(n)}/{max(n)}, {sum(x != 128 for x in n)} not exactly 128")
EOF
echo "$(date) ALL_DONE"
