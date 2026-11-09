#!/bin/bash
# Label the first 100,000 masked-SFT v2 train pairs with Qwen2.5-7B-Instruct on both GPUs (dynamic chunk claims).
# Relaunch with a NEW CLAIMS dir to resume (already-labeled pairs are skipped).
cd $(git rev-parse --show-toplevel)
PY=python
CLAIMS=data/sft/llm_labels_v1/claims_$(date +%Y%m%d_%H%M%S)
for g in 0 1; do
  mem=0.80; [ $g = 1 ] && mem=0.88
  CUDA_VISIBLE_DEVICES=$g VLLM_LOGGING_LEVEL=WARNING nohup $PY phase1/sft/label_specifics_llm.py --split train \
    --start 0 --limit 100000 --tag gpu$g --claim-dir $CLAIMS --gpu-mem $mem \
    >> outputs/phase1/llm_labels/label_100k_gpu$g.log 2>&1 &
  echo "gpu$g pid $!"
done
