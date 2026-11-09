# Phase 3 ensemble members: "<run name>|<flags>". Shared by train.sh and inference.sh.
RUNS=../runs_p3/all
COMMON="--ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train CL_dev IT_dev --dev --ctx window --seed 0"
# fields: name | technique | epochs | window
MEMBERS=(
  "lorma_w3072_s0|--peft lorma --rank 8 --alpha 16 --lr 5e-4|20|--max_len 3072 --train_stride 1536 --eval_stride 1536 --batch_size 3"
  "lorma_w2048_s0|--peft lorma --rank 8 --alpha 16 --lr 5e-4|20|--max_len 2048 --train_stride 1024 --eval_stride 1024 --batch_size 4"
  "full_mlp_la_w2048_s0|--peft none --lr 5e-5 --mlp --la_tau 1|10|--max_len 2048 --train_stride 1024 --eval_stride 1024 --batch_size 4"
)
