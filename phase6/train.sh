#!/usr/bin/env bash
# Phase 6 training: the 6 summarizers used by the cross-system MBR ensemble.
# Usage: GPU=0 PYTHON=python bash phase6/train.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GPU="${GPU:-0}"; PYTHON="${PYTHON:-python}"

source ./models.sh
for m in "${MEMBERS[@]}"; do
  name=${m%%|*}; flags=${m#*|}
  [ -f "$RUNS/$name/weights.pt" ] && { echo "skip $name"; continue; }
  $PYTHON train_summ.py $COMMON $flags --gpu "$GPU" --out "$RUNS/$name"
done
