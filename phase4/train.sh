#!/usr/bin/env bash
# Phase 4 training: the 7 members of the weighted ensemble.
# Usage: GPU=0 PYTHON=python bash phase4/train.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GPU="${GPU:-0}"; PYTHON="${PYTHON:-python}"

source ./models.sh
for m in "${MEMBERS[@]}"; do
  name=${m%%|*}; flags=${m#*|}
  [ -f "$RUNS/$name/classifier.pt" ] && { echo "skip $name"; continue; }
  $PYTHON train_cjpe.py $COMMON $flags --gpu "$GPU" --save --out "$RUNS/$name"
done
