#!/bin/bash
# Official submissions for Arch E and Arch F: CAD (alpha 0.5, short 32, beta 0.1) samples, seeds 1-8,
# T 0.7, top-p 0.9, rep 1.1, batch 128, 320 new tokens (same recipe as outputs/phase1/fact_decoding/archE),
# then penalty-aware MBR x8 with --loop-filter and the frozen Arch E rerank weights
# (outputs/phase1/mbr_rerank/archE_local_rerank8_tuning.json) = the local 0.8546 config of Arch E.
# Prompts whose 8 samples are all < 128 words get 8 fresh samples with 480 new tokens (as fix411 before).
# GPU 1 only.
set -e
export CUDA_VISIBLE_DEVICES=1
cd $(git rev-parse --show-toplevel)
PY=python
O=outputs/phase1/cad_official_EF
W=$($PY -c "import json;print(json.dumps(json.load(open('outputs/phase1/mbr_rerank/archE_local_rerank8_tuning.json'))['weights']))")
CAD="--temperature 0.7 --top-p 0.9 --repetition-penalty 1.1 --batch-size 128 --max-mem-gb 26 --cad --cad-alpha 0.5 --cad-short-len 32 --cad-beta 0.1"
echo "rerank weights: $W"
for m in E:runs/archE_diff_145M/ckpt_best.pt F:runs/archF_nodiff_145M/ckpt_best.pt; do
  name=${m%%:*}; ckpt=${m#*:}; D=$O/arch$name; mkdir -p $D
  for s in 1 2 3 4 5 6 7 8; do
    if [ ! -f $D/s$s/prediction.json ]; then
      $PY phase1/generate.py --ckpt $ckpt --prompts official $CAD --max-new-tokens 320 --seed $s --out $D/s$s > $D/s$s.log 2>&1
    fi
    echo "$(date +%T) $name seed $s done: $(grep -o '"seconds": [0-9.]*' $D/s$s/generation_stats.json) $(grep -o 'WARNING: [0-9]*' $D/s$s.log || echo 'all >=128 words')"
  done
  $PY phase1/mbr/mbr_rerank.py --cands $D/s{1,2,3,4,5,6,7,8}/prediction.json --prompts official --weights "$W" \
      --loop-filter --out $D/mbr8_cad_lf_rerank.json --batch-size 64 2>&1 | grep -E "wrote" || true
  $PY $O/fix_and_pack.py --model $name --ckpt $ckpt --weights "$W" --cad-flags "$CAD"
done
echo "$(date) ALL_DONE"
