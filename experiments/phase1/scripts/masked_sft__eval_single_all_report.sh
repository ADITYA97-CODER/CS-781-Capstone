#!/bin/bash
# After the 'none' screen (GPU 0) and the upweight seed-1 sample (GPU 1) exist: single-sample report over all 4 models.
cd $(git rev-parse --show-toplevel)
E=outputs/phase1/masked_sft/eval
until [ -f $E/sft_none/s1/prediction.json ] && [ -f $E/sft_upweight/s1/prediction.json ] \
      && ! pgrep -f "eval_masked_sft.py --stage single --gpus 0 --models start,sft_none" > /dev/null; do sleep 15; done
echo "$(date +%T) all four seed-1 samples present; report"
CUDA_VISIBLE_DEVICES=0 python phase1/sft/eval_masked_sft.py --stage single --gpus 0
rc=$?
echo "$(date +%T) SINGLE_ALL_DONE exit $rc"
