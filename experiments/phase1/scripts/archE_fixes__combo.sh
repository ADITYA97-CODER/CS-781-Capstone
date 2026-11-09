#!/bin/bash
# Step 1: score combinations that need no new samples. Step 2 (after the official run on GPU 0 finishes):
# generate Arch E CAD + steering samples (seeds 1-4) and score everything again. GPU 0.
set -e
export CUDA_VISIBLE_DEVICES=0
cd $(git rev-parse --show-toplevel)
PY="python"
O=outputs/phase1/archE_fixes
$PY $O/combo.py
echo "$(date +%T) step 1 done; waiting for the official run"
until grep -qE "ALL_DONE|Traceback" outputs/phase1/archE_steer_official/run.log; do sleep 30; done
for s in 1 2 3 4; do
  if [ ! -f $O/cad_steer/s$s/prediction.json ]; then
    mkdir -p $O/cad_steer
    $PY phase1/generate.py --ckpt runs/archE_diff_145M/ckpt_best.pt --prompts local --temperature 0.7 --top-p 0.9 \
        --repetition-penalty 1.1 --batch-size 128 --max-new-tokens 320 --max-mem-gb 26 \
        --cad --cad-alpha 0.5 --cad-short-len 32 --cad-beta 0.1 --steer-headers --steer-cite \
        --seed $s --out $O/cad_steer/s$s > $O/cad_steer/s$s.log 2>&1
  fi
  echo "$(date +%T) cad_steer seed $s done"
done
$PY $O/combo.py
echo "$(date) ALL_DONE"
