#!/bin/bash
# Masked task SFT: 3 runs (none / unsupported / unsupported+upweight) on GPUs 0 and 1. Each GPU pulls the next
# config from a shared queue (flock), so the third run starts on whichever GPU finishes first.
cd $(git rev-parse --show-toplevel)
Q=outputs/phase1/masked_sft/train_queue.txt
printf "%s\n" archE_sft_v2_unsupported archE_sft_v2_none archE_sft_v2_upweight > $Q
PY=python
worker() {
  gpu=$1
  while true; do
    cfg=$(flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$cfg" ] && break
    name=sft_${cfg#archE_sft_}; name=sft_archE_${cfg#archE_sft_}
    echo "$(date +%T) GPU $gpu start $cfg"
    CUDA_VISIBLE_DEVICES=$gpu $PY phase1/sft/train_sft.py --config phase1/configs/sft/$cfg.json > runs/$name.log 2>&1
    rc=$?
    echo "$(date +%T) GPU $gpu end $cfg exit $rc"
  done
}
worker 0 & sleep 20; worker 1 &
wait
echo "$(date +%T) ALL_TRAIN_DONE"
