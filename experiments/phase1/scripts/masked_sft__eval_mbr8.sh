#!/bin/bash
# 8-sample MBR evaluation (seed 1 reused): seeds 2-8 for all 4 models on GPUs 0,1, rerank, then REPORT.md.
cd $(git rev-parse --show-toplevel)
PY=python
$PY phase1/sft/eval_masked_sft.py --stage gen --gpus 0,1 || { echo "GEN_FAILED"; exit 1; }
CUDA_VISIBLE_DEVICES=0 $PY phase1/sft/eval_masked_sft.py --stage report
echo "$(date +%T) MBR8_DONE exit $?"
