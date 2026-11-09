# Phase 7 best model: Arch F + LoRMA+ bi-encoder, fused with lexical and metadata features.
RUN=f_lorma
RUNS=../runs_p7
FLAGS="--ckpt ../runs/archF_nodiff_145M/ckpt_best.pt --max_len 512 --max_chunks 8 --rows 8 --seq 2048 --bq 16 --n_hard 3 --hard_pool 30 --bm bm25bi --tau 0.05 --epochs 6 --evals_per_epoch 1 --peft lorma --rank 8 --alpha 16 --lr 5e-4 --doc_pool mean --seed 0"
