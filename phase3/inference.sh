#!/usr/bin/env bash
# Phase 3 inference: per-member test probabilities, mean-probability ensemble, Dissent rule.
# Usage: GPU=0 PYTHON=python bash phase3/inference.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GPU="${GPU:-0}"; PYTHON="${PYTHON:-python}"

source ./models.sh
members=()
for m in "${MEMBERS[@]}"; do
  IFS='|' read -r name tech epochs win <<< "$m"
  $PYTHON train_rr.py $COMMON $tech $win --epochs 0 --load "$RUNS/$name/classifier.pt" --predict --gpu "$GPU" --out "$RUNS/$name"
  members+=("$RUNS/$name")
done
$PYTHON make_final.py --members "${members[@]}" --lam 0 --out ../outputs/phase3/all_wide3
$PYTHON apply_dissent_rule.py --pred ../outputs/phase3/all_wide3/prediction.json --out ../outputs/phase3/final
