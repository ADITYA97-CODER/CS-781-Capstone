#!/usr/bin/env bash
# Phase 1 inference: pooled MBR x16 (Arch E x8 + masked-SFT x8) with steering and penalty-aware rerank.
# Usage: GPU=0 PYTHON=python bash phase1/inference.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GPU="${GPU:-0}"; PYTHON="${PYTHON:-python}"

cd ..
$PYTHON phase1/pooled_submission.py --gpu "$GPU"
