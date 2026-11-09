#!/bin/bash
# Wait for the v2 SFT training queue to finish, then run the single-sample screen (seed 1, all 4 models) on GPUs 0,1.
cd $(git rev-parse --show-toplevel)
until grep -q ALL_TRAIN_DONE outputs/phase1/masked_sft/train_all_v2.log; do sleep 30; done
for r in none unsupported upweight; do
  [ -f runs/sft_archE_v2_$r/ckpt_final.pt ] || { echo "run $r did not finish (no ckpt_final.pt)"; exit 1; }
done
echo "$(date +%T) training done; single-sample screen starting"
python phase1/sft/eval_masked_sft.py --stage single --gpus 0,1
echo "$(date +%T) SINGLE_DONE exit $?"
