#!/bin/bash
# Two-GPU version of run.sh (same flags): one worker per GPU pulls (model, seed) jobs from a shared queue
# (claimed atomically with mkdir), then MBR x8 + loop filter + frozen rerank and packing for both models.
# Arch E seed 1 was already being generated on GPU 1 by run.sh (PID given as $1); the GPU 1 worker waits for it.
set -e
cd $(git rev-parse --show-toplevel)
PY=python
O=outputs/phase1/cad_official_EF
WAIT_PID=${1:-}
W=$($PY -c "import json;print(json.dumps(json.load(open('outputs/phase1/mbr_rerank/archE_local_rerank8_tuning.json'))['weights']))")
CAD="--temperature 0.7 --top-p 0.9 --repetition-penalty 1.1 --batch-size 128 --max-mem-gb 26 --cad --cad-alpha 0.5 --cad-short-len 32 --cad-beta 0.1"
declare -A CKPT=([E]=runs/archE_diff_145M/ckpt_best.pt [F]=runs/archF_nodiff_145M/ckpt_best.pt)
worker() {
  local gpu=$1
  for name in E F; do for s in 1 2 3 4 5 6 7 8; do
    local D=$O/arch$name; mkdir -p $D
    [ -f $D/s$s/prediction.json ] && continue
    mkdir $D/s$s.claim 2>/dev/null || continue          # another worker has this job
    CUDA_VISIBLE_DEVICES=$gpu $PY phase1/generate.py --ckpt ${CKPT[$name]} --prompts official $CAD --max-new-tokens 320 \
        --seed $s --out $D/s$s > $D/s$s.log 2>&1
    echo "$(date +%T) GPU $gpu: $name seed $s done: $(grep -o '"seconds": [0-9.]*' $D/s$s/generation_stats.json) $(grep -o 'WARNING: [0-9]*' $D/s$s.log || echo 'all >=128 words')"
  done; done
}
mkdir -p $O/archE/s1.claim                               # in progress under run.sh
worker 0 & P0=$!
( if [ -n "$WAIT_PID" ]; then while kill -0 $WAIT_PID 2>/dev/null; do sleep 10; done;
    echo "$(date +%T) GPU 1: E seed 1 done (from run.sh): $(grep -o '"seconds": [0-9.]*' $O/archE/s1/generation_stats.json)"; fi
  worker 1 ) & P1=$!
wait $P0; wait $P1
for name in E F; do
  for s in 1 2 3 4 5 6 7 8; do [ -f $O/arch$name/s$s/prediction.json ] || { echo "MISSING $name s$s"; exit 1; }; done
done
echo "$(date +%T) all 16 sample sets done; MBR + rerank + packing"
for name in E F; do
  D=$O/arch$name
  CUDA_VISIBLE_DEVICES=1 $PY phase1/mbr/mbr_rerank.py --cands $D/s{1,2,3,4,5,6,7,8}/prediction.json --prompts official \
      --weights "$W" --loop-filter --out $D/mbr8_cad_lf_rerank.json --batch-size 64 2>&1 | grep -E "wrote" || true
  CUDA_VISIBLE_DEVICES=1 $PY $O/fix_and_pack.py --model $name --ckpt ${CKPT[$name]} --weights "$W" --cad-flags "$CAD"
done
echo "$(date) ALL_DONE"
