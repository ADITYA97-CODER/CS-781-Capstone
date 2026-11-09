#!/usr/bin/env bash
# Phase 3 training: three wide-context Arch E configurations on all 90 labelled documents.
# Usage: GPU=0 PYTHON=python bash phase3/train.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GPU="${GPU:-0}"; PYTHON="${PYTHON:-python}"

source ./models.sh
for m in "${MEMBERS[@]}"; do
  IFS='|' read -r name tech epochs win <<< "$m"
  [ -f "$RUNS/$name/classifier.pt" ] && { echo "skip $name"; continue; }
  $PYTHON train_rr.py $COMMON $tech $win --epochs "$epochs" --gpu "$GPU" --save --out "$RUNS/$name"
done
