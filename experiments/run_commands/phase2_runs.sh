# Recorded arguments of every Phase 2 run, taken from the saved run configurations (GPU omitted).
# Merge experiments first (see experiments/README.md), then run a line from the phase2 folder.
# A line with --epochs 0 is a prediction-only rerun of an already trained model.

# e_sftup_bitfit_all_s0
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 fold2 --predict fold3 --out ../runs_p2/e_sftup_bitfit_all_s0 --epochs 20 --batch_size 8 --lr 0.002 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft bitfit --rank 8 --alpha 16 --head bio --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --windows doc --ctx 64 --unit_len 256 --ctx_dropout 0.0 --crf_lr 0.01 --crf_ce 0.0 --kd_T 4.0 --kd_alpha 0.5 --seed 0 --save

# e_sftup_full_all_s0
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 fold2 --predict fold3 --out ../runs_p2/e_sftup_full_all_s0 --epochs 16 --batch_size 8 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --seed 0 --save

# e_sftup_full_all_s1
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 fold2 --predict fold3 --out ../runs_p2/e_sftup_full_all_s1 --epochs 16 --batch_size 8 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --seed 1 --save

# e_sftup_full_cv1
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 --dev fold2 --out ../runs_p2/e_sftup_full_cv1 --epochs 20 --batch_size 8 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --seed 0

# e_sftup_full_cv2
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold2 --dev fold1 --out ../runs_p2/e_sftup_full_cv2 --epochs 20 --batch_size 8 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --seed 0

# e_sftup_ia3_all_s0
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 fold2 --predict fold3 --out ../runs_p2/e_sftup_ia3_all_s0 --epochs 20 --batch_size 8 --lr 0.003 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft ia3 --rank 8 --alpha 16 --head bio --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --windows doc --ctx 64 --unit_len 256 --ctx_dropout 0.0 --crf_lr 0.01 --crf_ce 0.0 --kd_T 4.0 --kd_alpha 0.5 --seed 0 --save

# e_sftup_lora_all_s0
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 fold2 --predict fold3 --out ../runs_p2/e_sftup_lora_all_s0 --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lora --rank 8 --alpha 16.0 --head bio --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --windows doc --ctx 64 --unit_len 256 --ctx_dropout 0.0 --crf_lr 0.01 --crf_ce 0.0 --kd_T 4.0 --kd_alpha 0.5 --seed 0 --save

# e_sftup_lorma_all_s0
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 fold2 --predict fold3 --out ../runs_p2/e_sftup_lorma_all_s0 --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --seed 0 --save

# e_sftup_lorma_all_s1
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 fold2 --predict fold3 --out ../runs_p2/e_sftup_lorma_all_s1 --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --seed 1 --save

# e_sftup_lorma_all_s2
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 fold2 --predict fold3 --out ../runs_p2/e_sftup_lorma_all_s2 --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --head bio --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --windows doc --ctx 64 --unit_len 256 --ctx_dropout 0.0 --kd_T 4.0 --kd_alpha 0.5 --seed 2 --save

# e_sftup_lorma_all_s3
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 fold2 --predict fold3 --out ../runs_p2/e_sftup_lorma_all_s3 --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --head bio --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --windows doc --ctx 64 --unit_len 256 --ctx_dropout 0.0 --kd_T 4.0 --kd_alpha 0.5 --seed 3 --save

# e_sftup_lorma_crf_all_s0
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 fold2 --predict fold3 --out ../runs_p2/e_sftup_lorma_crf_all_s0 --epochs 12 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --head bio --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --windows doc --ctx 64 --unit_len 256 --ctx_dropout 0.0 --crf --crf_lr 0.001 --crf_ce 1.0 --kd_T 4.0 --kd_alpha 0.5 --seed 0 --save

# e_sftup_lorma_crf_all_s1
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 fold2 --predict fold3 --out ../runs_p2/e_sftup_lorma_crf_all_s1 --epochs 12 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --head bio --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --windows doc --ctx 64 --unit_len 256 --ctx_dropout 0.0 --crf --crf_lr 0.001 --crf_ce 1.0 --kd_T 4.0 --kd_alpha 0.5 --seed 1 --save

# e_sftup_lorma_crf_cv1
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 --dev fold2 --out ../runs_p2/e_sftup_lorma_crf_cv1 --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --head bio --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --windows doc --ctx 64 --unit_len 256 --ctx_dropout 0.0 --crf --crf_lr 0.001 --crf_ce 1.0 --kd_T 4.0 --kd_alpha 0.5 --seed 0

# e_sftup_lorma_cv1
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 --dev fold2 --out ../runs_p2/e_sftup_lorma_cv1 --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --seed 0

# e_sftup_lorma_cv2
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold2 --dev fold1 --out ../runs_p2/e_sftup_lorma_cv2 --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --seed 0

# e_sftup_lorma_cvlp_cv1
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 --dev fold2 --out ../runs_p2/e_sftup_lorma_cvlp_cv1 --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --head bio --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --windows doc --ctx 64 --unit_len 256 --ctx_dropout 0.0 --crf_lr 0.01 --crf_ce 0.0 --kd_T 4.0 --kd_alpha 0.5 --seed 0

# e_sftup_lorma_cvlp_cv2
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold2 --dev fold1 --out ../runs_p2/e_sftup_lorma_cvlp_cv2 --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --head bio --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --windows doc --ctx 64 --unit_len 256 --ctx_dropout 0.0 --crf_lr 0.01 --crf_ce 0.0 --kd_T 4.0 --kd_alpha 0.5 --seed 0

# e_sftup_lorma_flert128_all_s0
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 fold2 --predict fold3 --out ../runs_p2/e_sftup_lorma_flert128_all_s0 --epochs 20 --batch_size 16 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --head bio --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --windows flert --ctx 128 --unit_len 256 --ctx_dropout 0.0 --seed 0 --save

# e_sftup_lorma_flert_all_s0
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 fold2 --predict fold3 --out ../runs_p2/e_sftup_lorma_flert_all_s0 --epochs 20 --batch_size 16 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --head bio --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --windows flert --ctx 64 --unit_len 256 --ctx_dropout 0.0 --seed 0 --save

# e_sftup_lorma_flert_all_s1
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 fold2 --predict fold3 --out ../runs_p2/e_sftup_lorma_flert_all_s1 --epochs 20 --batch_size 16 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --head bio --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --windows flert --ctx 64 --unit_len 256 --ctx_dropout 0.0 --seed 1 --save

# e_sftup_lorma_flert_cv1
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 --dev fold2 --out ../runs_p2/e_sftup_lorma_flert_cv1 --epochs 20 --batch_size 16 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --head bio --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --windows flert --ctx 64 --unit_len 256 --ctx_dropout 0.0 --seed 0

# e_sftup_lorma_flert_cv2
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold2 --dev fold1 --out ../runs_p2/e_sftup_lorma_flert_cv2 --epochs 20 --batch_size 16 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --head bio --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --windows flert --ctx 64 --unit_len 256 --ctx_dropout 0.0 --seed 0

# e_sftup_lorma_kd_all_s0
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 fold2 --predict fold3 --out ../runs_p2/e_sftup_lorma_kd_all_s0 --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --head bio --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --windows doc --ctx 64 --unit_len 256 --ctx_dropout 0.0 --kd ../runs_p2/kd_teacher_bio4.npz --kd_T 4.0 --kd_alpha 0.5 --seed 0 --save

# e_sftup_lorma_kd_all_s1
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 fold2 --predict fold3 --out ../runs_p2/e_sftup_lorma_kd_all_s1 --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --head bio --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --windows doc --ctx 64 --unit_len 256 --ctx_dropout 0.0 --kd ../runs_p2/kd_teacher_bio4.npz --kd_T 4.0 --kd_alpha 0.5 --seed 1 --save

# e_sftup_lorma_span_all_s0
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 fold2 --predict fold3 --out ../runs_p2/e_sftup_lorma_span_all_s0 --epochs 0 --batch_size 8 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --head span --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --seed 0 --load ../runs_p2/e_sftup_lorma_span_all_s0/tagger_final.pt

# e_sftup_lorma_span_all_s1
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 fold2 --predict fold3 --out ../runs_p2/e_sftup_lorma_span_all_s1 --epochs 0 --batch_size 8 --lr 5e-05 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --head span --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --seed 1 --load ../runs_p2/e_sftup_lorma_span_all_s1/tagger_final.pt

# e_sftup_lorma_span_all_s2
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 fold2 --predict fold3 --out ../runs_p2/e_sftup_lorma_span_all_s2 --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --head span --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --windows doc --ctx 64 --unit_len 256 --ctx_dropout 0.0 --crf_lr 0.01 --crf_ce 0.0 --kd_T 4.0 --kd_alpha 0.5 --seed 2 --save

# e_sftup_lorma_span_all_s3
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 fold2 --predict fold3 --out ../runs_p2/e_sftup_lorma_span_all_s3 --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --head span --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --windows doc --ctx 64 --unit_len 256 --ctx_dropout 0.0 --crf_lr 0.01 --crf_ce 0.0 --kd_T 4.0 --kd_alpha 0.5 --seed 3 --save

# e_sftup_lorma_span_cv1
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold1 --dev fold2 --out ../runs_p2/e_sftup_lorma_span_cv1 --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --head span --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --seed 0

# e_sftup_lorma_span_cv2
python train_ner.py --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --train fold2 --dev fold1 --out ../runs_p2/e_sftup_lorma_span_cv2 --epochs 20 --batch_size 8 --lr 0.0005 --head_lr 0.001 --weight_decay 0.01 --warmup_frac 0.06 --dropout 0.1 --header_len 128 --content_len 1024 --train_stride 512 --eval_stride 768 --peft lorma --rank 8 --alpha 16.0 --head span --max_span 48 --span_hidden 256 --span_threshold 0.5 --span_neg_ratio 20.0 --seed 0

# combination: combo_bio4_span
python make_combo.py --bio_runs ../runs_p2/e_sftup_lorma_all_s0 ../runs_p2/e_sftup_lorma_all_s1 ../runs_p2/e_sftup_lorma_flert_all_s0 ../runs_p2/e_sftup_lorma_flert_all_s1 --span_runs ../runs_p2/e_sftup_lorma_span_all_s0 ../runs_p2/e_sftup_lorma_span_all_s1 --span_labels RESP STAT PREC CASENO --threshold 0.5 --out ../outputs/phase2/combo_bio4_span

# combination: combo_bio6_span2
python make_combo.py --bio_runs ../runs_p2/e_sftup_lorma_all_s0 ../runs_p2/e_sftup_lorma_all_s1 ../runs_p2/e_sftup_lorma_all_s2 ../runs_p2/e_sftup_lorma_all_s3 ../runs_p2/e_sftup_lorma_flert_all_s0 ../runs_p2/e_sftup_lorma_flert_all_s1 --span_runs ../runs_p2/e_sftup_lorma_span_all_s0 ../runs_p2/e_sftup_lorma_span_all_s1 --span_labels RESP STAT PREC CASENO --threshold 0.5 --out ../outputs/phase2/combo_bio6_span2

# combination: combo_bio6_span4
python make_combo.py --bio_runs ../runs_p2/e_sftup_lorma_all_s0 ../runs_p2/e_sftup_lorma_all_s1 ../runs_p2/e_sftup_lorma_all_s2 ../runs_p2/e_sftup_lorma_all_s3 ../runs_p2/e_sftup_lorma_flert_all_s0 ../runs_p2/e_sftup_lorma_flert_all_s1 --span_runs ../runs_p2/e_sftup_lorma_span_all_s0 ../runs_p2/e_sftup_lorma_span_all_s1 ../runs_p2/e_sftup_lorma_span_all_s2 ../runs_p2/e_sftup_lorma_span_all_s3 --span_labels RESP STAT PREC CASENO --threshold 0.5 --out ../outputs/phase2/combo_bio6_span4

# combination: combo_lorma_bio_span
python make_combo.py --bio_runs ../runs_p2/e_sftup_lorma_all_s0 ../runs_p2/e_sftup_lorma_all_s1 --span_runs ../runs_p2/e_sftup_lorma_span_all_s0 ../runs_p2/e_sftup_lorma_span_all_s1 --span_labels RESP STAT PREC CASENO --threshold 0.5 --out ../outputs/phase2/combo_lorma_bio_span

# combination: combo_lorma_flert_span
python make_combo.py --bio_runs ../runs_p2/e_sftup_lorma_flert_all_s0 ../runs_p2/e_sftup_lorma_flert_all_s1 --span_runs ../runs_p2/e_sftup_lorma_span_all_s0 ../runs_p2/e_sftup_lorma_span_all_s1 --span_labels RESP STAT PREC CASENO --threshold 0.5 --out ../outputs/phase2/combo_lorma_flert_span
