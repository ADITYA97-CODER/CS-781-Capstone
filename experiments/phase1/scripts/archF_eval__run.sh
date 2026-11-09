#!/bin/bash
# Arch F vs Arch E, single sample (seed 1), same recipe as outputs/phase1/mbr_archE/run.sh, both regenerated
# back to back on GPU 1 so GPU nondeterminism cannot favour either model. Report: outputs/phase1/archF_eval/compare.py
set -e
export CUDA_VISIBLE_DEVICES=1
cd $(git rev-parse --show-toplevel)
PY=python
O=outputs/phase1/archF_eval
for m in F:runs/archF_nodiff_145M/ckpt_best.pt E:runs/archE_diff_145M/ckpt_best.pt; do
  name=${m%%:*}; ckpt=${m#*:}
  $PY phase1/generate.py --ckpt $ckpt --prompts local --temperature 0.7 --top-p 0.9 --repetition-penalty 1.1 \
      --seed 1 --batch-size 128 --out $O/${name}_s1 > $O/${name}_s1.log 2>&1
  echo "$(date +%T) generated $name seed 1"
done
echo ALL_DONE
