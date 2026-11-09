#!/usr/bin/env bash
# Phase 2 training: 6 LoRMA+ BIO taggers and 2 LoRMA+ span heads on folds 1 + 2.
# Usage: GPU=0 PYTHON=python bash phase2/train.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GPU="${GPU:-0}"; PYTHON="${PYTHON:-python}"

source ./models.sh
for m in "${MEMBERS[@]}"; do
  name=${m%%|*}; flags=${m#*|}
  [ -f "$RUNS/$name/tagger_final.pt" ] && { echo "skip $name"; continue; }
  $PYTHON train_ner.py $COMMON $flags --epochs 20 --gpu "$GPU" --save --out "$RUNS/$name"
done
