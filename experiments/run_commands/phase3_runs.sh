# Recorded arguments of every Phase 3 run, taken from the saved run configurations (GPU omitted).
# Merge experiments first (see experiments/README.md), then run a line from the phase3 folder.
# A line with --epochs 0 is a prediction-only rerun of an already trained model.

# all/full_mlp_la_w2048_s0
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train CL_dev IT_dev --dev --predict --out ../runs_p3/all/full_mlp_la_w2048_s0 --epochs 10 --batch_size 4 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 2048 --train_stride 1024 --eval_stride 1024 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --mlp --la_tau 1.0 --peft none --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 0

# all/full_mlp_la_w2048_s1
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train CL_dev IT_dev --dev --predict --out ../runs_p3/all/full_mlp_la_w2048_s1 --epochs 10 --batch_size 4 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 2048 --train_stride 1024 --eval_stride 1024 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --mlp --la_tau 1.0 --peft none --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 1

# all/full_mlp_la_w2048_s2
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train CL_dev IT_dev --dev --predict --out ../runs_p3/all/full_mlp_la_w2048_s2 --epochs 10 --batch_size 4 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 2048 --train_stride 1024 --eval_stride 1024 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --mlp --la_tau 1.0 --peft none --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 2

# all/laya_nm_s0
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train CL_dev IT_dev --dev --predict --out ../runs_p3/all/laya_nm_s0 --epochs 10 --stop_epoch 4 --batch_size 8 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --mlp --la_tau 1.0 --peft none --rank 8 --alpha 16 --seed 0

# all/laya_nm_s1
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train CL_dev IT_dev --dev --predict --out ../runs_p3/all/laya_nm_s1 --epochs 10 --stop_epoch 4 --batch_size 8 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --mlp --la_tau 1.0 --peft none --rank 8 --alpha 16 --seed 1

# all/laya_nm_s2
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train CL_dev IT_dev --dev --predict --out ../runs_p3/all/laya_nm_s2 --epochs 10 --stop_epoch 4 --batch_size 8 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --mlp --la_tau 1.0 --peft none --rank 8 --alpha 16 --seed 2

# all/laya_nm_s3
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train CL_dev IT_dev --dev --predict --out ../runs_p3/all/laya_nm_s3 --epochs 10 --stop_epoch 4 --batch_size 8 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --mlp --la_tau 1.0 --peft none --rank 8 --alpha 16 --seed 3

# all/lorma_mlp_la_w2048_s0
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train CL_dev IT_dev --dev --predict --out ../runs_p3/all/lorma_mlp_la_w2048_s0 --epochs 20 --batch_size 4 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 2048 --train_stride 1024 --eval_stride 1024 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --mlp --la_tau 1.0 --peft lorma --rank 8 --alpha 16.0 --kd_T 2.0 --kd_alpha 0.5 --seed 0

# all/lorma_w2048_s0
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train CL_dev IT_dev --dev --predict --out ../runs_p3/all/lorma_w2048_s0 --epochs 20 --batch_size 4 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 2048 --train_stride 1024 --eval_stride 1024 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft lorma --rank 8 --alpha 16.0 --kd_T 2.0 --kd_alpha 0.5 --seed 0

# all/lorma_w2048_s1
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train CL_dev IT_dev --dev --predict --out ../runs_p3/all/lorma_w2048_s1 --epochs 20 --batch_size 4 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 2048 --train_stride 1024 --eval_stride 1024 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft lorma --rank 8 --alpha 16.0 --kd_T 2.0 --kd_alpha 0.5 --seed 1

# all/lorma_w3072_s0
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train CL_dev IT_dev --dev --predict --out ../runs_p3/all/lorma_w3072_s0 --epochs 20 --batch_size 3 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 3072 --train_stride 1536 --eval_stride 1536 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft lorma --rank 8 --alpha 16.0 --kd_T 2.0 --kd_alpha 0.5 --seed 0

# all/lorma_w3072_s1
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train CL_dev IT_dev --dev --predict --out ../runs_p3/all/lorma_w3072_s1 --epochs 20 --batch_size 3 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 3072 --train_stride 1536 --eval_stride 1536 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft lorma --rank 8 --alpha 16.0 --kd_T 2.0 --kd_alpha 0.5 --seed 1

# all/lorma_w3072_s2
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train CL_dev IT_dev --dev --predict --out ../runs_p3/all/lorma_w3072_s2 --epochs 20 --batch_size 3 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 3072 --train_stride 1536 --eval_stride 1536 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft lorma --rank 8 --alpha 16.0 --kd_T 2.0 --kd_alpha 0.5 --seed 2

# cv/ctx_none_nopos_f0
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train cv2_f0 --dev cv2_f1 --out ../runs_p3/cv/ctx_none_nopos_f0 --epochs 20 --batch_size 32 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx none --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --bucket --n_pos 1 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft lorma --rank 8 --alpha 16.0 --seed 0 --epoch_logp

# cv/ctx_none_nopos_f1
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train cv2_f1 --dev cv2_f0 --out ../runs_p3/cv/ctx_none_nopos_f1 --epochs 20 --batch_size 32 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx none --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --bucket --n_pos 1 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft lorma --rank 8 --alpha 16.0 --seed 0 --epoch_logp

# cv/ctx_none_pos_f0
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train cv2_f0 --dev cv2_f1 --out ../runs_p3/cv/ctx_none_pos_f0 --epochs 20 --batch_size 32 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx none --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --bucket --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft lorma --rank 8 --alpha 16.0 --seed 0 --epoch_logp

# cv/ctx_none_pos_f1
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train cv2_f1 --dev cv2_f0 --out ../runs_p3/cv/ctx_none_pos_f1 --epochs 20 --batch_size 32 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx none --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --bucket --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft lorma --rank 8 --alpha 16.0 --seed 0 --epoch_logp

# cv/ctx_win1024_f0
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train cv2_f0 --dev cv2_f1 --out ../runs_p3/cv/ctx_win1024_f0 --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft lorma --rank 8 --alpha 16.0 --seed 0 --epoch_logp

# cv/ctx_win1024_f1
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train cv2_f1 --dev cv2_f0 --out ../runs_p3/cv/ctx_win1024_f1 --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft lorma --rank 8 --alpha 16.0 --seed 0 --epoch_logp

# cv/ctx_win256_f0
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train cv2_f0 --dev cv2_f1 --out ../runs_p3/cv/ctx_win256_f0 --epochs 20 --batch_size 32 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 256 --train_stride 128 --eval_stride 128 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft lorma --rank 8 --alpha 16.0 --seed 0 --epoch_logp

# cv/ctx_win256_f1
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train cv2_f1 --dev cv2_f0 --out ../runs_p3/cv/ctx_win256_f1 --epochs 20 --batch_size 32 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 256 --train_stride 128 --eval_stride 128 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft lorma --rank 8 --alpha 16.0 --seed 0 --epoch_logp

# cv/full_mlp_la_f0
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train cv2_f0 --dev cv2_f1 --out ../runs_p3/cv/full_mlp_la_f0 --epochs 10 --batch_size 8 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --mlp --la_tau 1.0 --peft none --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --epoch_logp --epoch_test_logp

# cv/full_mlp_la_f1
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train cv2_f1 --dev cv2_f0 --out ../runs_p3/cv/full_mlp_la_f1 --epochs 10 --batch_size 8 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --mlp --la_tau 1.0 --peft none --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --epoch_logp --epoch_test_logp

# cv/full_mlp_la_w2048_f0
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train cv2_f0 --dev cv2_f1 --out ../runs_p3/cv/full_mlp_la_w2048_f0 --epochs 10 --batch_size 4 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 2048 --train_stride 1024 --eval_stride 1024 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --mlp --la_tau 1.0 --peft none --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --epoch_logp --epoch_test_logp

# cv/full_mlp_la_w2048_f1
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train cv2_f1 --dev cv2_f0 --out ../runs_p3/cv/full_mlp_la_w2048_f1 --epochs 10 --batch_size 4 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 2048 --train_stride 1024 --eval_stride 1024 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --mlp --la_tau 1.0 --peft none --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --epoch_logp --epoch_test_logp

# cv/lorma_mlp_la_f0
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train cv2_f0 --dev cv2_f1 --out ../runs_p3/cv/lorma_mlp_la_f0 --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --mlp --la_tau 1.0 --peft lorma --rank 8 --alpha 16.0 --seed 0 --epoch_logp

# cv/lorma_mlp_la_f1
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train cv2_f1 --dev cv2_f0 --out ../runs_p3/cv/lorma_mlp_la_f1 --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --mlp --la_tau 1.0 --peft lorma --rank 8 --alpha 16.0 --seed 0 --epoch_logp

# cv/lorma_mlp_la_w2048_f0
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train cv2_f0 --dev cv2_f1 --out ../runs_p3/cv/lorma_mlp_la_w2048_f0 --epochs 20 --batch_size 4 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 2048 --train_stride 1024 --eval_stride 1024 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --mlp --la_tau 1.0 --peft lorma --rank 8 --alpha 16.0 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --epoch_logp --epoch_test_logp

# cv/lorma_mlp_la_w2048_f1
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train cv2_f1 --dev cv2_f0 --out ../runs_p3/cv/lorma_mlp_la_w2048_f1 --epochs 20 --batch_size 4 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 2048 --train_stride 1024 --eval_stride 1024 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --mlp --la_tau 1.0 --peft lorma --rank 8 --alpha 16.0 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --epoch_logp --epoch_test_logp

# cv/lorma_win2048_f0
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train cv2_f0 --dev cv2_f1 --out ../runs_p3/cv/lorma_win2048_f0 --epochs 20 --batch_size 4 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 2048 --train_stride 1024 --eval_stride 1024 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft lorma --rank 8 --alpha 16.0 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --epoch_logp --epoch_test_logp

# cv/lorma_win2048_f1
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train cv2_f1 --dev cv2_f0 --out ../runs_p3/cv/lorma_win2048_f1 --epochs 20 --batch_size 4 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 2048 --train_stride 1024 --eval_stride 1024 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft lorma --rank 8 --alpha 16.0 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --epoch_logp --epoch_test_logp

# cv/lorma_win3072_f0
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train cv2_f0 --dev cv2_f1 --out ../runs_p3/cv/lorma_win3072_f0 --epochs 20 --batch_size 3 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 3072 --train_stride 1536 --eval_stride 1536 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft lorma --rank 8 --alpha 16.0 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --epoch_logp --epoch_test_logp

# cv/lorma_win3072_f1
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train cv2_f1 --dev cv2_f0 --out ../runs_p3/cv/lorma_win3072_f1 --epochs 20 --batch_size 3 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 3072 --train_stride 1536 --eval_stride 1536 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft lorma --rank 8 --alpha 16.0 --kd_T 2.0 --kd_alpha 0.5 --seed 0 --epoch_logp --epoch_test_logp

# e10_kd_ctx
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train --dev CL_dev IT_dev --predict --out ../runs_p3/e10_kd_ctx --epochs 10 --batch_size 8 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft none --rank 8 --alpha 16 --n_layers 10 --kd ../runs_p3/cv/ctx_win1024:17 ../runs_p3/cv/lorma_mlp_la:7 --kd_T 2.0 --kd_alpha 0.5 --seed 0

# e10_nokd_ctx
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train --dev CL_dev IT_dev --predict --out ../runs_p3/e10_nokd_ctx --epochs 10 --batch_size 8 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft none --rank 8 --alpha 16 --n_layers 10 --kd_T 2.0 --kd_alpha 0.5 --seed 0

# e_full_ctx
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train --dev CL_dev IT_dev --predict --out ../runs_p3/e_full_ctx --epochs 10 --batch_size 8 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --peft none --rank 8 --alpha 16 --seed 0

# e_full_laya
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train --dev CL_dev IT_dev --predict --out ../runs_p3/e_full_laya --epochs 10 --batch_size 8 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head laya --la_tau 1.0 --peft none --rank 8 --alpha 16 --seed 0

# e_full_laya_nomarker
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train --dev CL_dev IT_dev --predict --out ../runs_p3/e_full_laya_nomarker --epochs 10 --batch_size 8 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --mlp --la_tau 1.0 --peft none --rank 8 --alpha 16 --seed 0

# e_ia3_ctx
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train --dev CL_dev IT_dev --predict --out ../runs_p3/e_ia3_ctx --epochs 20 --batch_size 8 --lr 0.003 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft ia3 --rank 8 --alpha 16 --kd_T 2.0 --kd_alpha 0.5 --seed 0

# e_lora_ctx
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train --dev CL_dev IT_dev --predict --out ../runs_p3/e_lora_ctx --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft lora --rank 8 --alpha 16.0 --seed 0

# e_lorma_ctx
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train --dev CL_dev IT_dev --predict --out ../runs_p3/e_lorma_ctx --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft lorma --rank 8 --alpha 16.0 --seed 0

# e_lorma_mlp_la
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train --dev CL_dev IT_dev --predict --out ../runs_p3/e_lorma_mlp_la --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --mlp --la_tau 1.0 --peft lorma --rank 8 --alpha 16.0 --seed 0

# e_rslora_ctx
python train_rr.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train CL_train IT_train --dev CL_dev IT_dev --predict --out ../runs_p3/e_rslora_ctx --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --ctx window --max_len 1024 --train_stride 512 --eval_stride 512 --max_sent_tok 192 --n_pos 16 --class_weight none --label_smoothing 0.0 --head pool --la_tau 0.0 --peft rslora --rank 64 --alpha 16.0 --kd_T 2.0 --kd_alpha 0.5 --seed 0

# ensemble: all_wide3
python make_final.py --members ../runs_p3/all/lorma_w3072_s0 ../runs_p3/all/lorma_w2048_s0 ../runs_p3/all/full_mlp_la_w2048_s0 --weights 0.333 0.333 0.333 --lam 0.0 --out ../outputs/phase3/all_wide3

# ensemble: all_wide3_5m
python make_final.py --members ../runs_p3/all/lorma_w3072_s0 ../runs_p3/all/lorma_w3072_s1 ../runs_p3/all/lorma_w2048_s0 ../runs_p3/all/full_mlp_la_w2048_s0 ../runs_p3/all/full_mlp_la_w2048_s1 --weights 0.167 0.167 0.333 0.167 0.167 --lam 0.0 --out ../outputs/phase3/all_wide3_5m

# ensemble: all_wide3_9m
python make_final.py --members ../runs_p3/all/lorma_w3072_s0 ../runs_p3/all/lorma_w3072_s1 ../runs_p3/all/lorma_w3072_s2 ../runs_p3/all/lorma_w2048_s0 ../runs_p3/all/lorma_w2048_s1 ../runs_p3/all/full_mlp_la_w2048_s0 ../runs_p3/all/full_mlp_la_w2048_s1 ../runs_p3/all/full_mlp_la_w2048_s2 --weights 0.111 0.111 0.111 0.167 0.167 0.111 0.111 0.111 --lam 0.0 --out ../outputs/phase3/all_wide3_9m

# ensemble: cvhalves_wide3
python make_final.py --members ../runs_p3/cv/lorma_win3072:20 ../runs_p3/cv/lorma_win2048:20 ../runs_p3/cv/full_mlp_la_w2048:10 --weights 0.333 0.333 0.333 --lam 0.0 --out ../outputs/phase3/cvhalves_wide3
