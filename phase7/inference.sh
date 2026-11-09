#!/usr/bin/env bash
# Phase 7 inference: embed dev + test, build lexical features, fit the two-stage LTR, write the top 7.
# Usage: GPU=0 PYTHON=python bash phase7/inference.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GPU="${GPU:-0}"; PYTHON="${PYTHON:-python}"

source ./models.sh
$PYTHON train_pcr.py $FLAGS --gpu "$GPU" --inference --out "$RUNS/$RUN"
$PYTHON ltr.py --dense "$RUN" --fit stack --seeds 3 --isq_shift 0.71 --out ../outputs/phase7/f_lorma
