#!/usr/bin/env bash
# Phase 1 training: corpus preparation, Arch E and Arch F pretraining, masked task SFT.
# Usage: GPU=0 PYTHON=python bash phase1/train.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GPU="${GPU:-0}"; PYTHON="${PYTHON:-python}"

cd ..                                   # Phase-1 scripts run from the code root
export CUDA_VISIBLE_DEVICES="$GPU"

# 1. Corpus: normalize the four sources, filter and split (also writes the 990-prompt local eval set).
$PYTHON phase1/corpus/normalize.py
$PYTHON phase1/corpus/filter_split.py

# 2. Tokenizer: artifacts/tokenizer/legal_bpe_32k is included. To retrain it instead:
#    $PYTHON phase1/tokenizer/train_tokenizer.py --vocab-size 32000 --sample-words 300000000 \
#        --mix supreme_court=0.25,high_courts=0.40,tribunals=0.20,acts=0.15 --min-frequency 5 --seed 42
$PYTHON phase1/corpus/tokenize_corpus.py --tokenizer legal_bpe_32k

# 3. Pretraining (4B tokens each). Arch E is the Phase-1 model; Arch F is the backbone of the Arch F runs in Phases 4-7.
$PYTHON phase1/train.py --config phase1/configs/archE.json
$PYTHON phase1/train.py --config phase1/configs/archF.json

# 4. Masked task SFT: clean the corpus, build prompt -> 128-word pairs and replay text, fine-tune Arch E.
$PYTHON phase1/corpus/write_cleaned_corpus.py
$PYTHON phase1/sft/heldout_stats.py
$PYTHON phase1/sft/build_pairs.py --n-train 50000 --n-val 1000 --out data/sft/masked_sft_v1
$PYTHON phase1/sft/build_pairs.py --n-train 150000 --out data/sft/masked_sft_v2 --reuse-val data/sft/masked_sft_v1
$PYTHON phase1/sft/build_replay.py
$PYTHON phase1/sft/train_sft.py --config phase1/configs/sft/archE_sft_v2_upweight.json
