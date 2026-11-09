#!/bin/bash
# Official submission: 4 LLaMA samples (t=0.7, top-p=0.9, rep 1.1) on the official test prompts + MBR (GPU 0 only).
set -e
export CUDA_VISIBLE_DEVICES=0
cd $(git rev-parse --show-toplevel)
PY="python"
O=outputs/llama_phase1/mbr_official
for s in 1 2 3 4; do
  $PY phase1/mbr/llama_sample.py --prompts official --seed $s --out $O/llama_s$s.json > $O/llama_s$s.log 2>&1
  echo "sample seed $s done"
done
$PY phase1/mbr/mbr_select.py --cands $O/llama_s{1,2,3,4}.json --out $O/prediction.json 2>&1 | grep -E "picked|consensus|wrote|Error|Traceback"
echo ALL_DONE
