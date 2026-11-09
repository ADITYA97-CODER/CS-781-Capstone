# Recorded arguments of every Phase 5 run, taken from the saved run configurations (GPU omitted).
# Merge experiments first (see experiments/README.md), then run a line from the phase5 folder.
# A line with --epochs 0 is a prediction-only rerun of an already trained model.

# e_full_spread8
python train_lsi.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p5/e_full_spread8 --view spread --max_len 1024 --max_chunks 8 --head_len 256 --rows 4 --seq 4096 --accum 2 --epochs 3.0 --evals_per_epoch 2 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.1 --doc_pool attn --n_pos 8 --head pool --loss bce --asl_gamma_neg 4.0 --asl_clip 0.05 --peft none --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --save

# e_lorma_lwan
python train_lsi.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p5/e_lorma_lwan --view spread --max_len 1024 --max_chunks 8 --head_len 256 --rows 4 --seq 4096 --accum 2 --epochs 3.0 --evals_per_epoch 2 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.1 --doc_pool attn --n_pos 8 --head lwan --loss bce --asl_gamma_neg 4.0 --asl_clip 0.05 --peft lorma --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --save

# e_lorma_spread8
python train_lsi.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p5/e_lorma_spread8 --view spread --max_len 1024 --max_chunks 8 --head_len 256 --rows 4 --seq 4096 --accum 2 --epochs 3.0 --evals_per_epoch 2 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.1 --doc_pool attn --n_pos 8 --loss bce --asl_gamma_neg 4.0 --asl_clip 0.05 --peft lorma --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --save

# e_lwan_asl
python train_lsi.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p5/e_lwan_asl --view spread --max_len 1024 --max_chunks 8 --head_len 256 --rows 4 --seq 4096 --accum 2 --epochs 5.0 --evals_per_epoch 2 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.1 --doc_pool attn --n_pos 8 --head lwan --loss asl --asl_gamma_neg 4.0 --asl_clip 0.05 --peft lorma --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --save

# f_lorma_spread8
python train_lsi.py --ckpt ../runs/archF_nodiff_145M/ckpt_best.pt --out ../runs_p5/f_lorma_spread8 --view spread --max_len 1024 --max_chunks 8 --head_len 256 --rows 4 --seq 4096 --accum 2 --epochs 3.0 --evals_per_epoch 2 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.1 --doc_pool attn --n_pos 8 --head pool --loss bce --asl_gamma_neg 4.0 --asl_clip 0.05 --peft lorma --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --save

# ensemble: ens5_lwan2
python make_ensemble.py --runs ../runs_p5/e_lorma_lwan ../runs_p5/e_lwan_asl ../runs_p5/e_full_spread8 ../runs_p5/f_lorma_spread8 ../runs_p5/e_lorma_spread8 --weights 0.2857 0.2857 0.1429 0.1429 0.1429 --out ../outputs/phase5/ens5_lwan2
