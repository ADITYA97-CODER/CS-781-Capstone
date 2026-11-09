#!/bin/bash
# MBR experiment on the local held-out set (GPU 0 only).
set -e
export CUDA_VISIBLE_DEVICES=0
cd $(git rev-parse --show-toplevel)
PY="python"
M=outputs/phase1/mbr
EVAL() { echo "== $1"; $PY "official_eval/Evaluation Script/evaluate_phase1.py" --truth data/corpus/eval/local_ref.json --prediction $2 2>/dev/null | grep -E "BERT|ROUGE|BLEU"; }
for s in 1 2 3 4; do
  $PY phase1/generate.py --ckpt runs/archD_sliding_72M/ckpt_best.pt --prompts local --temperature 0.7 --top-p 0.9 \
      --repetition-penalty 1.1 --seed $s --batch-size 128 --out $M/archD_s$s > $M/archD_s$s.log 2>&1
  cp $M/archD_s$s/prediction.json $M/archD_s$s.json
  EVAL "archD sample seed $s" $M/archD_s$s.json
done
for s in 1 2 3 4; do
  $PY phase1/mbr/llama_sample.py --prompts local --seed $s --out $M/llama_s$s.json > $M/llama_s$s.log 2>&1
  EVAL "llama sample seed $s" $M/llama_s$s.json
done
$PY phase1/mbr/mbr_select.py --cands $M/archD_s{1,2,3,4}.json --out $M/mbr_archD4.json
EVAL "MBR archD x4" $M/mbr_archD4.json
$PY phase1/mbr/mbr_select.py --cands $M/llama_s{1,2,3,4}.json --out $M/mbr_llama4.json
EVAL "MBR llama x4" $M/mbr_llama4.json
$PY phase1/mbr/mbr_select.py --cands $M/archD_s{1,2,3,4}.json $M/llama_s{1,2,3,4}.json --out $M/mbr_both8.json
EVAL "MBR both x8" $M/mbr_both8.json
echo ALL_DONE
