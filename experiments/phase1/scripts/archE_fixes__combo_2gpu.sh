#!/bin/bash
# Arch E CAD + steering samples (seeds 1-2 on GPU 0, seeds 3-4 on GPU 1, in parallel), then combo.py (GPU 0).
cd $(git rev-parse --show-toplevel)
PY="python"
O=outputs/phase1/archE_fixes
mkdir -p $O/cad_steer
rm -f $O/cad_steer/s1.log                          # partial log of the stopped run (no output was written)
gen() {  # gpu seed...
  local g=$1; shift
  for s in "$@"; do
    if [ ! -f $O/cad_steer/s$s/prediction.json ]; then
      CUDA_VISIBLE_DEVICES=$g $PY phase1/generate.py --ckpt runs/archE_diff_145M/ckpt_best.pt --prompts local \
          --temperature 0.7 --top-p 0.9 --repetition-penalty 1.1 --batch-size 128 --max-new-tokens 320 --max-mem-gb 26 \
          --cad --cad-alpha 0.5 --cad-short-len 32 --cad-beta 0.1 --steer-headers --steer-cite \
          --seed $s --out $O/cad_steer/s$s > $O/cad_steer/s$s.log 2>&1 || echo "seed $s FAILED (GPU $g)"
    fi
    echo "$(date +%T) cad_steer seed $s done (GPU $g)"
  done
}
gen 0 1 2 &
gen 1 3 4 &
wait
CUDA_VISIBLE_DEVICES=0 $PY $O/combo.py
echo "$(date) ALL_DONE"
