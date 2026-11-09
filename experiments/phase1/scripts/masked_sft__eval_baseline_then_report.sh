#!/bin/bash
# After the upweight seeds finish: Arch E steer seeds 5-8 (seeds 1-4 already in eval/start), rerank, report (start vs upweight).
cd $(git rev-parse --show-toplevel)
PY=python
while pgrep -f "eval_masked_sft.py --stage gen --gpus 0,1 --models sft_upweight" > /dev/null; do sleep 20; done
grep -q "rerank sft_upweight" outputs/phase1/masked_sft/eval_mbr8_upweight.log || { echo "upweight gen/rerank did not finish"; exit 1; }
echo "$(date +%T) upweight done; baseline seeds 5-8"
$PY phase1/sft/eval_masked_sft.py --stage gen --gpus 0,1 --models start || { echo "BASELINE_GEN_FAILED"; exit 1; }
CUDA_VISIBLE_DEVICES=0 $PY phase1/sft/eval_masked_sft.py --stage report
echo "$(date +%T) MBR8_REPORT_DONE exit $?"
