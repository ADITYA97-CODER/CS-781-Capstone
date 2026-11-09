# Recorded arguments of every Phase 4 run, taken from the saved run configurations (GPU omitted).
# Merge experiments first (see experiments/README.md), then run a line from the phase4 folder.
# A line with --epochs 0 is a prediction-only rerun of an already trained model.

# e_full_hier4x1k
python train_cjpe.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p4/e_full_hier4x1k --view hier --max_len 1024 --max_chunks 4 --head_len 256 --rows 4 --seq 4096 --accum 2 --epochs 2.0 --evals_per_epoch 2 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.1 --doc_pool attn --n_pos 8 --class_weight none --peft none --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --save

# e_full_hier8x1k
python train_cjpe.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p4/e_full_hier8x1k --view hier --max_len 1024 --max_chunks 8 --head_len 256 --rows 4 --seq 4096 --accum 2 --epochs 2.0 --evals_per_epoch 2 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.1 --doc_pool attn --n_pos 8 --class_weight none --peft none --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --save

# e_lora_hier4x1k
python train_cjpe.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p4/e_lora_hier4x1k --view hier --max_len 1024 --max_chunks 4 --head_len 256 --rows 4 --seq 4096 --accum 2 --epochs 2.0 --evals_per_epoch 2 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.1 --doc_pool attn --n_pos 8 --class_weight none --peft lora --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --save

# e_lorma_first2k
python train_cjpe.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p4/e_lorma_first2k --view first --max_len 2048 --max_chunks 1 --head_len 256 --rows 8 --seq 2048 --accum 2 --epochs 2.0 --evals_per_epoch 2 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.1 --doc_pool mean --n_pos 8 --class_weight none --peft lorma --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --save

# e_lorma_hier4x1k
python train_cjpe.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p4/e_lorma_hier4x1k --view hier --max_len 1024 --max_chunks 4 --head_len 256 --rows 4 --seq 4096 --accum 2 --epochs 2.0 --evals_per_epoch 2 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.1 --doc_pool attn --n_pos 8 --class_weight none --peft lorma --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --save

# e_lorma_hier8x1k
python train_cjpe.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p4/e_lorma_hier8x1k --view hier --max_len 1024 --max_chunks 8 --head_len 256 --rows 4 --seq 4096 --accum 2 --epochs 2.0 --evals_per_epoch 2 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.1 --doc_pool attn --n_pos 8 --class_weight none --peft lorma --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --save

# e_lorma_last2k
python train_cjpe.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p4/e_lorma_last2k --view last --max_len 2048 --max_chunks 1 --head_len 256 --rows 8 --seq 2048 --accum 2 --epochs 2.0 --evals_per_epoch 2 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.1 --doc_pool mean --n_pos 8 --class_weight none --peft lorma --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --save

# e_partial6_hier4x1k
python train_cjpe.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p4/e_partial6_hier4x1k --view hier --max_len 1024 --max_chunks 4 --head_len 256 --rows 4 --seq 4096 --accum 2 --epochs 2.0 --evals_per_epoch 2 --lr 0.0001 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.1 --doc_pool attn --n_pos 8 --class_weight none --peft none --rank 8 --alpha 16 --train_layers 6 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --save

# f_full_hier4x1k
python train_cjpe.py --ckpt ../runs/archF_nodiff_145M/ckpt_best.pt --out ../runs_p4/f_full_hier4x1k --view hier --max_len 1024 --max_chunks 4 --head_len 256 --rows 4 --seq 4096 --accum 2 --epochs 2.0 --evals_per_epoch 2 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.1 --doc_pool attn --n_pos 8 --class_weight none --peft none --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --save

# f_lorma_hier4x1k
python train_cjpe.py --ckpt ../runs/archF_nodiff_145M/ckpt_best.pt --out ../runs_p4/f_lorma_hier4x1k --view hier --max_len 1024 --max_chunks 4 --head_len 256 --rows 4 --seq 4096 --accum 2 --epochs 2.0 --evals_per_epoch 2 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.1 --doc_pool attn --n_pos 8 --class_weight none --peft lorma --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --save

# f_lorma_hier8x1k
python train_cjpe.py --ckpt ../runs/archF_nodiff_145M/ckpt_best.pt --out ../runs_p4/f_lorma_hier8x1k --view hier --max_len 1024 --max_chunks 8 --head_len 256 --rows 4 --seq 4096 --accum 2 --epochs 2.0 --evals_per_epoch 2 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.1 --doc_pool attn --n_pos 8 --class_weight none --peft lorma --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --save

# f_lorma_last2k
python train_cjpe.py --ckpt ../runs/archF_nodiff_145M/ckpt_best.pt --out ../runs_p4/f_lorma_last2k --view last --max_len 2048 --max_chunks 1 --head_len 256 --rows 8 --seq 2048 --accum 2 --epochs 2.0 --evals_per_epoch 2 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.1 --doc_pool mean --n_pos 8 --class_weight none --peft lorma --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --save

# ensemble: ens6_all
python make_ensemble.py --runs ../runs_p4/e_lorma_hier4x1k ../runs_p4/e_full_hier4x1k ../runs_p4/e_lora_hier4x1k ../runs_p4/f_lorma_hier4x1k ../runs_p4/e_lorma_last2k ../runs_p4/f_lorma_last2k --weights 0.16666666666666666 0.16666666666666666 0.16666666666666666 0.16666666666666666 0.16666666666666666 0.16666666666666666 --out ../outputs/phase4/ens6_all

# ensemble: ens7_with8
python make_ensemble.py --runs ../runs_p4/e_lorma_hier8x1k ../runs_p4/e_lorma_hier4x1k ../runs_p4/e_full_hier4x1k ../runs_p4/e_lora_hier4x1k ../runs_p4/f_lorma_hier4x1k ../runs_p4/f_lorma_last2k ../runs_p4/e_lorma_last2k --weights 0.14285714285714285 0.14285714285714285 0.14285714285714285 0.14285714285714285 0.14285714285714285 0.14285714285714285 0.14285714285714285 --out ../outputs/phase4/ens7_with8

# ensemble: ens7_with8_w3
python make_ensemble.py --runs ../runs_p4/e_lorma_hier8x1k ../runs_p4/e_lorma_hier4x1k ../runs_p4/e_full_hier4x1k ../runs_p4/e_lora_hier4x1k ../runs_p4/f_lorma_hier4x1k ../runs_p4/f_lorma_last2k ../runs_p4/e_lorma_last2k --weights 0.3333333333333333 0.1111111111111111 0.1111111111111111 0.1111111111111111 0.1111111111111111 0.1111111111111111 0.1111111111111111 --out ../outputs/phase4/ens7_with8_w3
