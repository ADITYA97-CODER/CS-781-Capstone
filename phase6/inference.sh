#!/usr/bin/env bash
# Phase 6 inference: beam-4 generation with every summarizer, then cross-system MBR selection.
# Usage: GPU=0 PYTHON=python bash phase6/inference.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GPU="${GPU:-0}"; PYTHON="${PYTHON:-python}"

source ./models.sh
preds=()
for m in "${MEMBERS[@]}"; do
  name=${m%%|*}
  [ -f "$RUNS/$name/gen_test_b4_nr4/prediction.json" ] || \
    $PYTHON gen_summ.py --run "$RUNS/$name" --split test --beams 4 --no_repeat 4 --tag b4_nr4 --gpu "$GPU"
  preds+=("$RUNS/$name/gen_test_b4_nr4/prediction.json")
done
CUDA_VISIBLE_DEVICES="$GPU" $PYTHON mbr_select.py --split test --preds "${preds[@]}" --out ../outputs/phase6/mbr6_b4_test
