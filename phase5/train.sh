#!/usr/bin/env bash
# Phase 5 training: the 5 members of the weighted ensemble.
# Usage: GPU=0 PYTHON=python bash phase5/train.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GPU="${GPU:-0}"; PYTHON="${PYTHON:-python}"

source ./models.sh
for m in "${MEMBERS[@]}"; do
  name=${m%%|*}; flags=${m#*|}
  [ -f "$RUNS/$name/classifier.pt" ] && { echo "skip $name"; continue; }
  $PYTHON train_lsi.py $COMMON $flags --gpu "$GPU" --save --out "$RUNS/$name"
done
