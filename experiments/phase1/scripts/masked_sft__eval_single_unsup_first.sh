#!/bin/bash
# As soon as the 'unsupported' SFT run finishes on GPU 0: single-sample screen (seed 1) for the start model and
# SFT-unsupported on GPU 0 (GPU 1 keeps training 'upweight'). Outputs are reused by the full screen later.
cd $(git rev-parse --show-toplevel)
until grep -q "end archE_sft_v2_unsupported" outputs/phase1/masked_sft/train_all_v2.log; do sleep 20; done
[ -f runs/sft_archE_v2_unsupported/ckpt_final.pt ] || { echo "unsupported run did not finish"; exit 1; }
echo "$(date +%T) unsupported done; screening start + sft_unsupported on GPU 0"
CUDA_VISIBLE_DEVICES=0 python phase1/sft/eval_masked_sft.py --stage single --gpus 0 --models start,sft_unsupported
rc=$?
echo "$(date +%T) SINGLE_UNSUP_DONE exit $rc"
