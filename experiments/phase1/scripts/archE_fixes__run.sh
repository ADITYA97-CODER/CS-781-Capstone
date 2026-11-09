#!/bin/bash
# Arch E decoding fixes, local prompts, seeds 1-4, same sampling as outputs/phase1/mbr_archE (GPU 0).
#   steer      = --steer-headers --steer-cite             (fix 2a + fix 3)
#   steer_close= --steer-headers --steer-cite --steer-close  (fix 2a + 2b + 3)
# Waits for the Arch E + CAD run to finish first. Then compare.py (MBR x4, +rerank, paired metrics).
set -e
export CUDA_VISIBLE_DEVICES=0
cd $(git rev-parse --show-toplevel)
PY="python"
CKPT=runs/archE_diff_145M/ckpt_best.pt
O=outputs/phase1/archE_fixes
while pgrep -f "outputs/phase1/fact_decoding/archE/run.sh" >/dev/null; do sleep 30; done
echo "$(date +%T) CAD run finished; starting fixes"
declare -A FLAGS=( [steer]="--steer-headers --steer-cite" [steer_close]="--steer-headers --steer-cite --steer-close" )
for cfg in steer steer_close; do
  for s in 1 2 3 4; do
    if [ ! -f $O/$cfg/s$s/prediction.json ]; then
      mkdir -p $O/$cfg
      $PY phase1/generate.py --ckpt $CKPT --prompts local --temperature 0.7 --top-p 0.9 --repetition-penalty 1.1 \
          --batch-size 128 --max-new-tokens 320 --max-mem-gb 26 ${FLAGS[$cfg]} --seed $s --out $O/$cfg/s$s > $O/$cfg/s$s.log 2>&1
    fi
    echo "$(date +%T) $cfg seed $s done: $(grep -o '"seconds": [0-9.]*' $O/$cfg/s$s/generation_stats.json) $(grep -o 'WARNING: [0-9]*' $O/$cfg/s$s.log || echo 'all >=128 words')"
  done
done
$PY $O/compare.py
echo "$(date) ALL_DONE"
