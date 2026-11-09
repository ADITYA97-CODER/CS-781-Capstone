#!/bin/bash
# CAD (alpha 0.5, short_len 32, beta 0.1) on Arch E, local prompts, seeds 1-8, same sampling as the
# Arch E baseline in outputs/phase1/mbr_archE (T 0.7, top-p 0.9, rep 1.1, batch 128). GPU 0.
# Then MBR x4 / x8 and the official scorer; compare.py adds paired metrics vs the baseline samples.
set -e
export CUDA_VISIBLE_DEVICES=0
cd $(git rev-parse --show-toplevel)
PY="python"
CKPT=runs/archE_diff_145M/ckpt_best.pt
M=outputs/phase1/fact_decoding/archE/cad_a0.5
EVAL() { echo "== $1"; $PY "official_eval/Evaluation Script/evaluate_phase1.py" --truth data/corpus/eval/local_ref.json --prediction $2 2>/dev/null | grep -E "BERT|ROUGE|BLEU"; }
mkdir -p $M
for s in 1 2 3 4 5 6 7 8; do
  if [ ! -f $M/s$s/prediction.json ]; then
    $PY phase1/generate.py --ckpt $CKPT --prompts local --temperature 0.7 --top-p 0.9 --repetition-penalty 1.1 \
        --batch-size 128 --max-new-tokens 320 --max-mem-gb 26 --cad --cad-alpha 0.5 --cad-short-len 32 --cad-beta 0.1 \
        --seed $s --out $M/s$s > $M/s$s.log 2>&1
  fi
  echo "$(date +%T) seed $s done: $(grep -o '"seconds": [0-9.]*' $M/s$s/generation_stats.json) $(grep -o 'WARNING: [0-9]*' $M/s$s.log || echo 'all >=128 words')"
  if [ $s = 4 ]; then
    $PY phase1/mbr/mbr_select.py --cands $M/s{1,2,3,4}/prediction.json --out $M/mbr4.json --batch-size 64 2>&1 | grep -E "consensus"
    EVAL "Arch E + CAD, MBR x4" $M/mbr4.json
  fi
done
EVAL "Arch E + CAD, single sample seed 1" $M/s1/prediction.json
$PY phase1/mbr/mbr_select.py --cands $M/s{1,2,3,4,5,6,7,8}/prediction.json --out $M/mbr8.json --batch-size 64 2>&1 | grep -E "consensus"
EVAL "Arch E + CAD, MBR x8" $M/mbr8.json
$PY outputs/phase1/fact_decoding/archE/compare.py
echo "$(date) ALL_DONE"
