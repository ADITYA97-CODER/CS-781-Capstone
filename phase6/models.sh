# Phase 6 ensemble members: "<run name>|<flags>". Shared by train.sh and inference.sh. Order matters: the KD student needs e_lorma_ht4k.
RUNS=../runs_p6
COMMON="--view headtail --doc_len 4096 --head_len 3072 --sum_len 1536 --len_tag 1 --rows 4 --seq 6144 --accum 4 --epochs 4 --evals_per_epoch 2 --dropout 0.0 --seed 0"
LORMA="--peft lorma --rank 8 --alpha 16 --lr 5e-4"
MEMBERS=(
  "e_lorma_ht4k|--arch E $LORMA"
  "f_lorma_ht4k|--arch F $LORMA"
  "e_houlsby_ht4k|--arch E --peft houlsby --adapter_size 64 --lr 5e-4"
  "e_full_ht4k|--arch E --peft none --lr 5e-5"
  "e_lora_ht4k|--arch E --peft lora --rank 8 --alpha 16 --lr 5e-4"
  "kd_e10_from_lorma|--arch E --peft none --lr 1e-4 --n_layers 10 --kd $RUNS/e_lorma_ht4k --kd_T 1 --kd_alpha 0.5"
)
