#!/usr/bin/env bash
# Phase 4 inference: dev + test logits per member, then the weighted ensemble with a dev-tuned threshold.
# Usage: GPU=0 PYTHON=python bash phase4/inference.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GPU="${GPU:-0}"; PYTHON="${PYTHON:-python}"

source ./models.sh
runs=()
for m in "${MEMBERS[@]}"; do
  name=${m%%|*}; flags=${m#*|}
  $PYTHON train_cjpe.py $COMMON $flags --gpu "$GPU" --inference --out "$RUNS/$name"
  runs+=("$RUNS/$name")
done
$PYTHON make_ensemble.py --runs "${runs[@]}" --weights $WEIGHTS --out ../outputs/phase4/ens7_with8_w3
