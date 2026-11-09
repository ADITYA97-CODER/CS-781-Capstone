#!/usr/bin/env bash
# Phase 2 inference: predict fold 3 with every member, then combine per label.
# Usage: GPU=0 PYTHON=python bash phase2/inference.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GPU="${GPU:-0}"; PYTHON="${PYTHON:-python}"

source ./models.sh
for m in "${MEMBERS[@]}"; do
  name=${m%%|*}; flags=${m#*|}
  $PYTHON train_ner.py $COMMON $flags --epochs 0 --load "$RUNS/$name/tagger_final.pt" --predict fold3 --gpu "$GPU" --out "$RUNS/$name"
done
mkdir -p ../outputs/phase2
$PYTHON make_combo.py --bio_runs "${BIO[@]/#/$RUNS/}" --span_runs "${SPANS[@]/#/$RUNS/}" \
    --span_labels RESP STAT PREC CASENO --threshold 0.5 --out ../outputs/phase2/combo_bio6_span2
