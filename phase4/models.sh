# Phase 4 ensemble members: "<run name>|<flags>". Shared by train.sh and inference.sh.
RUNS=../runs_p4
COMMON="--epochs 2 --evals_per_epoch 2 --accum 2 --seed 0"
LORMA="--peft lorma --rank 8 --alpha 16 --lr 5e-4"
HIER4="--view hier --max_len 1024 --max_chunks 4 --rows 4 --seq 4096 --doc_pool attn"
HIER8="--view hier --max_len 1024 --max_chunks 8 --rows 4 --seq 4096 --doc_pool attn"
LAST2K="--view last --max_len 2048 --max_chunks 1 --rows 8 --seq 2048 --doc_pool mean"
E="--ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt"
F="--ckpt ../runs/archF_nodiff_145M/ckpt_best.pt"
MEMBERS=(
  "e_lorma_hier8x1k|$E $LORMA $HIER8"
  "e_lorma_hier4x1k|$E $LORMA $HIER4"
  "e_full_hier4x1k|$E --peft none --lr 5e-5 $HIER4"
  "e_lora_hier4x1k|$E --peft lora --rank 8 --alpha 16 --lr 5e-4 $HIER4"
  "f_lorma_hier4x1k|$F $LORMA $HIER4"
  "f_lorma_last2k|$F $LORMA $LAST2K"
  "e_lorma_last2k|$E $LORMA $LAST2K"
)
WEIGHTS="3 1 1 1 1 1 1"
