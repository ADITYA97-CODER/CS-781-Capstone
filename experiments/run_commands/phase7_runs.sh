# Recorded arguments of every Phase 7 run, taken from the saved run configurations (GPU omitted).
# Merge experiments first (see experiments/README.md), then run a line from the phase7 folder.
# A line with --epochs 0 is a prediction-only rerun of an already trained model.

# ce_e_lorma
python rerank_ce.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p7/ce_e_lorma --q_len 1024 --c_win 512 --rows 8 --seq 2050 --n_neg 7 --neg_pool 30 --rerank 30 --accum 4 --epochs 1.5 --evals_per_epoch 2 --lr 0.0005 --head_lr 0.001 --warmup_frac 0.05 --peft lorma --rank 8 --alpha 16 --seed 0

# e_full
python train_pcr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p7/e_full --max_len 512 --max_chunks 8 --rows 8 --seq 2048 --bq 16 --n_hard 3 --hard_pool 30 --bm bm25bi --tau 0.05 --epochs 6.0 --evals_per_epoch 1 --lr 2e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --doc_pool mean --peft none --rank 8 --alpha 16 --n_prompt 0 --kd_T 1.0 --kd_alpha 0.5 --seed 0

# e_ia3
python train_pcr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p7/e_ia3 --max_len 512 --max_chunks 8 --rows 8 --seq 2048 --bq 16 --n_hard 3 --hard_pool 30 --bm bm25bi --tau 0.05 --epochs 6.0 --evals_per_epoch 1 --lr 0.003 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --doc_pool mean --peft ia3 --rank 8 --alpha 16 --n_prompt 0 --kd_T 1.0 --kd_alpha 0.5 --seed 0

# e_lorma
python train_pcr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p7/e_lorma --max_len 512 --max_chunks 8 --rows 8 --seq 2048 --bq 16 --n_hard 3 --hard_pool 30 --bm bm25bi --tau 0.05 --epochs 6.0 --evals_per_epoch 1 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --doc_pool mean --peft lorma --rank 8 --alpha 16 --n_prompt 0 --kd_T 1.0 --kd_alpha 0.5 --seed 0 --train_scores

# e_lorma_ctx
python train_pcr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p7/e_lorma_ctx --max_len 512 --max_chunks 8 --rows 8 --seq 2048 --bq 16 --n_hard 3 --hard_pool 30 --bm bm25bi --tau 0.05 --epochs 6.0 --evals_per_epoch 1 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --doc_pool mean --peft lorma --rank 8 --alpha 16 --n_prompt 0 --kd_T 1.0 --kd_alpha 0.5 --seed 0 --train_scores --query_view ctx

# e_partial6
python train_pcr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p7/e_partial6 --max_len 512 --max_chunks 8 --rows 8 --seq 2048 --bq 16 --n_hard 3 --hard_pool 30 --bm bm25bi --tau 0.05 --epochs 6.0 --evals_per_epoch 1 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --doc_pool mean --peft none --rank 8 --alpha 16 --n_prompt 0 --train_layers 6 --kd_T 1.0 --kd_alpha 0.5 --seed 0 --query_view full

# e_prompt
python train_pcr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p7/e_prompt --max_len 512 --max_chunks 8 --rows 8 --seq 2048 --bq 16 --n_hard 3 --hard_pool 30 --bm bm25bi --tau 0.05 --epochs 6.0 --evals_per_epoch 1 --lr 2e-05 --head_lr 0.01 --weight_decay 0.01 --warmup_frac 0.05 --doc_pool mean --peft prompt --rank 8 --alpha 16 --n_prompt 16 --kd_T 1.0 --kd_alpha 0.5 --seed 0 --query_view full

# f_lorma
python train_pcr.py --ckpt ../runs/archF_nodiff_145M/ckpt_best.pt --out ../runs_p7/f_lorma --max_len 512 --max_chunks 8 --rows 8 --seq 2048 --bq 16 --n_hard 3 --hard_pool 30 --bm bm25bi --tau 0.05 --epochs 6.0 --evals_per_epoch 1 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --doc_pool mean --peft lorma --rank 8 --alpha 16 --n_prompt 0 --kd_T 1.0 --kd_alpha 0.5 --seed 0

# f_lorma_ctx
python train_pcr.py --ckpt ../runs/archF_nodiff_145M/ckpt_best.pt --out ../runs_p7/f_lorma_ctx --max_len 512 --max_chunks 8 --rows 8 --seq 2048 --bq 16 --n_hard 3 --hard_pool 30 --bm bm25bi --tau 0.05 --epochs 6.0 --evals_per_epoch 1 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --doc_pool mean --peft lorma --rank 8 --alpha 16 --n_prompt 0 --kd_T 1.0 --kd_alpha 0.5 --seed 0 --query_view ctx

# kd_e10_from_lorma
python train_pcr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p7/kd_e10_from_lorma --max_len 512 --max_chunks 8 --rows 8 --seq 2048 --bq 16 --n_hard 3 --hard_pool 30 --bm bm25bi --tau 0.05 --epochs 6.0 --evals_per_epoch 1 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --doc_pool mean --peft none --rank 8 --alpha 16 --n_prompt 0 --n_layers 10 --kd ../runs_p7/e_lorma --kd_T 1.0 --kd_alpha 0.5 --seed 0 --query_view full

# learning-to-rank: ce_e_lorma
python ltr.py --dense --ce ce_e_lorma --dense_view doc --fit stack --drop --cand 150 --C 1.0 --folds 5 --seeds 3 --isq_shift 0.71 --out ../outputs/phase7/subs/ce_e_lorma

# learning-to-rank: e_full
python ltr.py --dense e_full --ce --dense_view doc --fit stack --drop --cand 150 --C 1.0 --folds 5 --seeds 3 --isq_shift 0.71 --out ../outputs/phase7/subs/e_full

# learning-to-rank: e_ia3
python ltr.py --dense e_ia3 --ce --dense_view doc --fit stack --drop --cand 150 --C 1.0 --folds 5 --seeds 3 --isq_shift 0.71 --out ../outputs/phase7/subs/e_ia3

# learning-to-rank: e_lorma
python ltr.py --dense e_lorma --ce --dense_view doc --fit stack --drop --cand 150 --C 1.0 --folds 5 --seeds 3 --isq_shift 0.71 --out ../outputs/phase7/subs/e_lorma

# learning-to-rank: e_lorma_ctx
python ltr.py --dense e_lorma_ctx --ce --dense_view doc --fit stack --drop --cand 150 --C 1.0 --folds 5 --seeds 3 --isq_shift 0.71 --out ../outputs/phase7/subs/e_lorma_ctx

# learning-to-rank: e_partial6
python ltr.py --dense e_partial6 --ce --dense_view doc --fit stack --drop --cand 150 --C 1.0 --folds 5 --seeds 3 --isq_shift 0.71 --out ../outputs/phase7/subs/e_partial6

# learning-to-rank: e_prompt
python ltr.py --dense e_prompt --ce --dense_view doc --fit stack --drop --cand 150 --C 1.0 --folds 5 --seeds 3 --isq_shift 0.71 --out ../outputs/phase7/subs/e_prompt

# learning-to-rank: ens_all_ce
python ltr.py --dense e_lorma f_lorma e_lorma_ctx f_lorma_ctx e_full kd_e10_from_lorma --ce ce_e_lorma --dense_view doc --fit stack --drop --cand 150 --C 1.0 --folds 5 --seeds 3 --isq_shift 0.71 --out ../outputs/phase7/subs/ens_all_ce

# learning-to-rank: ens_ctx_ce
python ltr.py --dense e_lorma_ctx f_lorma_ctx --ce ce_e_lorma --dense_view doc --fit stack --drop --cand 150 --C 1.0 --folds 5 --seeds 3 --isq_shift 0.71 --out ../outputs/phase7/subs/ens_ctx_ce

# learning-to-rank: ens_ctx_ce_content
python ltr.py --dense e_lorma_ctx f_lorma_ctx --ce ce_e_lorma --dense_view doc --fit stack --drop citator cite_text --cand 150 --C 1.0 --folds 5 --seeds 3 --isq_shift 0.71 --out ../outputs/phase7/subs/ens_ctx_ce_content

# learning-to-rank: f_lorma
python ltr.py --dense f_lorma --ce --dense_view doc --fit stack --drop --cand 150 --C 1.0 --folds 5 --seeds 3 --isq_shift 0.71 --out ../outputs/phase7/subs/f_lorma

# learning-to-rank: f_lorma_ctx
python ltr.py --dense f_lorma_ctx --ce --dense_view doc --fit stack --drop --cand 150 --C 1.0 --folds 5 --seeds 3 --isq_shift 0.71 --out ../outputs/phase7/subs/f_lorma_ctx

# learning-to-rank: kd_e10
python ltr.py --dense kd_e10_from_lorma --ce --dense_view doc --fit stack --drop --cand 150 --C 1.0 --folds 5 --seeds 3 --isq_shift 0.71 --out ../outputs/phase7/subs/kd_e10
