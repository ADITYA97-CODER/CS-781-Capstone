#!/usr/bin/env bash
# Phase 5 inference: dev + test logits per member, then the weighted ensemble with per-label thresholds.
# Usage: GPU=0 PYTHON=python bash phase5/inference.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GPU="${GPU:-0}"; PYTHON="${PYTHON:-python}"

source ./models.sh
runs=()
for m in "${MEMBERS[@]}"; do
  name=${m%%|*}; flags=${m#*|}
  $PYTHON train_lsi.py $COMMON $flags --gpu "$GPU" --inference --out "$RUNS/$name"
  runs+=("$RUNS/$name")
done
$PYTHON make_ensemble.py --runs "${runs[@]}" --weights $WEIGHTS --out ../outputs/phase5/ens5_lwan2
