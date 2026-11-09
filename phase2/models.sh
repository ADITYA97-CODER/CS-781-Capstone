# Phase 2 ensemble members: "<run name>|<flags>". Shared by train.sh and inference.sh.
RUNS=../runs_p2
COMMON="--ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 fold2 --peft lorma --rank 8 --alpha 16 --lr 5e-4"
FLERT="--windows flert --ctx 64 --unit_len 256 --batch_size 16"
SPAN="--head span --max_span 48 --span_neg_ratio 20 --span_threshold 0.5"
BIO=(e_sftup_lorma_all_s0 e_sftup_lorma_all_s1 e_sftup_lorma_all_s2 e_sftup_lorma_all_s3 e_sftup_lorma_flert_all_s0 e_sftup_lorma_flert_all_s1)
SPANS=(e_sftup_lorma_span_all_s0 e_sftup_lorma_span_all_s1)
MEMBERS=(
  "e_sftup_lorma_all_s0|--seed 0"
  "e_sftup_lorma_all_s1|--seed 1"
  "e_sftup_lorma_all_s2|--seed 2"
  "e_sftup_lorma_all_s3|--seed 3"
  "e_sftup_lorma_flert_all_s0|$FLERT --seed 0"
  "e_sftup_lorma_flert_all_s1|$FLERT --seed 1"
  "e_sftup_lorma_span_all_s0|$SPAN --seed 0"
  "e_sftup_lorma_span_all_s1|$SPAN --seed 1"
)
