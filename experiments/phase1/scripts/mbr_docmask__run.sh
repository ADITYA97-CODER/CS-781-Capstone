#!/bin/bash
# MBR over 8 sampled continuations of the doc-masked Arch D checkpoint (GPU 0 only).
set -e
export CUDA_VISIBLE_DEVICES=0
cd $(git rev-parse --show-toplevel)
PY="python"
CKPT=runs/archD_4B_docmask/ckpt_best.pt
M=outputs/phase1/mbr_docmask
EVAL() { echo "== $1"; $PY "official_eval/Evaluation Script/evaluate_phase1.py" --truth data/corpus/eval/local_ref.json --prediction $2 2>/dev/null | grep -E "BERT|ROUGE|BLEU"; }
for set in local official; do
  for s in 1 2 3 4 5 6 7 8; do
    $PY phase1/generate.py --ckpt $CKPT --prompts $set --temperature 0.7 --top-p 0.9 --repetition-penalty 1.1 \
        --seed $s --batch-size 128 --out $M/${set}_s$s > $M/${set}_s$s.log 2>&1
    echo "generated $set seed $s: $(grep -c . $M/${set}_s$s/prediction.json) lines; $(grep WARNING $M/${set}_s$s.log || echo 'all >=128 words')"
  done
done
EVAL "local single sample seed 1" $M/local_s1/prediction.json
$PY phase1/mbr/mbr_select.py --cands $M/local_s{1,2,3,4}/prediction.json --out $M/local_mbr4.json 2>&1 | grep -E "picked|consensus"
EVAL "local MBR x4" $M/local_mbr4.json
$PY phase1/mbr/mbr_select.py --cands $M/local_s{1,2,3,4,5,6,7,8}/prediction.json --out $M/local_mbr8.json 2>&1 | grep -E "picked|consensus"
EVAL "local MBR x8" $M/local_mbr8.json
mkdir -p $M/official
$PY phase1/mbr/mbr_select.py --cands $M/official_s{1,2,3,4,5,6,7,8}/prediction.json --out $M/official/prediction.json 2>&1 | grep -E "picked|consensus"
echo ALL_DONE
