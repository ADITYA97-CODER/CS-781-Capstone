#!/bin/bash
# Facts ceiling test: Run F (GPU 0) and Run C (GPU 1) in parallel, then the seed-1 screen on both GPUs.
cd $(git rev-parse --show-toplevel)
PY=python
L=outputs/phase1/facts_ceiling
CUDA_VISIBLE_DEVICES=0 $PY phase1/sft/train_sft.py --config phase1/configs/sft/archE_sft_facts_F.json > $L/train_F.log 2>&1 &
pF=$!
CUDA_VISIBLE_DEVICES=1 $PY phase1/sft/train_sft.py --config phase1/configs/sft/archE_sft_facts_C.json > $L/train_C.log 2>&1 &
pC=$!
wait $pF; rF=$?
wait $pC; rC=$?
echo "$(date '+%F %T') train F rc=$rF C rc=$rC"
if [ $rF -ne 0 ] || [ $rC -ne 0 ]; then echo "training failed; no screen"; exit 1; fi
$PY phase1/sft/eval_facts_ceiling.py --stage single --gpus 0,1 > $L/screen_single.log 2>&1
rc=$?
echo "$(date '+%F %T') screen rc=$rc"
