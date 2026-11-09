# Phase 5 ensemble members: "<run name>|<flags>". Shared by train.sh and inference.sh.
RUNS=../runs_p5
COMMON="--view spread --max_len 1024 --max_chunks 8 --rows 4 --seq 4096 --accum 2 --evals_per_epoch 2 --doc_pool attn --seed 0"
LORMA="--peft lorma --rank 8 --alpha 16 --lr 5e-4"
E="--ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt"
F="--ckpt ../runs/archF_nodiff_145M/ckpt_best.pt"
MEMBERS=(
  "e_lorma_lwan|$E $LORMA --head lwan --loss bce --epochs 3"
  "e_lwan_asl|$E $LORMA --head lwan --loss asl --asl_gamma_neg 4 --asl_clip 0.05 --epochs 5"
  "e_full_spread8|$E --peft none --lr 5e-5 --head pool --loss bce --epochs 3"
  "f_lorma_spread8|$F $LORMA --head pool --loss bce --epochs 3"
  "e_lorma_spread8|$E $LORMA --head pool --loss bce --epochs 3"
)
WEIGHTS="2 2 1 1 1"
