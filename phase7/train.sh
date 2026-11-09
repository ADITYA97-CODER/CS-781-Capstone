#!/usr/bin/env bash
# Phase 7 training: BM25 hard-negative scores, then the dense bi-encoder.
# Usage: GPU=0 PYTHON=python bash phase7/train.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GPU="${GPU:-0}"; PYTHON="${PYTHON:-python}"

source ./models.sh
mkdir -p "$RUNS/lex"
$PYTHON make_lex.py --name bm25bi --ngram 2 2 --min_df 2 --k1 4 --b 0.95
$PYTHON train_pcr.py $FLAGS --gpu "$GPU" --save --out "$RUNS/$RUN"
