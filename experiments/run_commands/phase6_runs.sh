# Recorded arguments of every Phase 6 run, taken from the saved run configurations (GPU omitted).
# Merge experiments first (see experiments/README.md), then run a line from the phase6 folder.
# A line with --epochs 0 is a prediction-only rerun of an already trained model.

# e_full_ht4k
python train_summ.py --arch E --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p6/e_full_ht4k --view headtail --doc_len 4096 --head_len 3072 --sum_len 1536 --len_tag 1 --rows 4 --seq 6144 --accum 4 --epochs 4.0 --evals_per_epoch 2 --lr 5e-05 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.0 --peft none --rank 8 --alpha 16 --adapter_size 64 --kd_T 1.0 --kd_alpha 0.5 --seed 0

# e_houlsby_ht4k
python train_summ.py --arch E --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p6/e_houlsby_ht4k --view headtail --doc_len 4096 --head_len 3072 --sum_len 1536 --len_tag 1 --rows 4 --seq 6144 --accum 4 --epochs 4.0 --evals_per_epoch 2 --lr 0.0005 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.0 --peft houlsby --rank 8 --alpha 16 --adapter_size 64 --kd_T 1.0 --kd_alpha 0.5 --seed 0

# e_lora_ht4k
python train_summ.py --arch E --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p6/e_lora_ht4k --view headtail --doc_len 4096 --head_len 3072 --sum_len 1536 --len_tag 1 --rows 4 --seq 6144 --accum 4 --epochs 4.0 --evals_per_epoch 2 --lr 0.0005 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.0 --peft lora --rank 8 --alpha 16.0 --adapter_size 64 --kd_T 1.0 --kd_alpha 0.5 --seed 0

# e_lorma_ht4k
python train_summ.py --arch E --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p6/e_lorma_ht4k --view headtail --doc_len 4096 --head_len 3072 --sum_len 1536 --len_tag 1 --rows 4 --seq 6144 --accum 4 --epochs 4.0 --evals_per_epoch 2 --lr 0.0005 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.0 --peft lorma --rank 8 --alpha 16 --adapter_size 64 --kd_T 1.0 --kd_alpha 0.5 --seed 0

# f_lorma_ht4k
python train_summ.py --arch F --ckpt ../runs/archF_nodiff_145M/ckpt_best.pt --out ../runs_p6/f_lorma_ht4k --view headtail --doc_len 4096 --head_len 3072 --sum_len 1536 --len_tag 1 --rows 4 --seq 6144 --accum 4 --epochs 4.0 --evals_per_epoch 2 --lr 0.0005 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.0 --peft lorma --rank 8 --alpha 16 --adapter_size 64 --kd_T 1.0 --kd_alpha 0.5 --seed 0

# kd_e10_from_lorma
python train_summ.py --arch E --ckpt ../runs/sft_archE_v2_upweight/ckpt_best.pt --out ../runs_p6/kd_e10_from_lorma --view headtail --doc_len 4096 --head_len 3072 --sum_len 1536 --len_tag 1 --rows 4 --seq 6144 --accum 4 --epochs 4.0 --evals_per_epoch 2 --lr 0.0001 --weight_decay 0.01 --warmup_frac 0.05 --dropout 0.0 --peft none --rank 8 --alpha 16 --adapter_size 64 --n_layers 10 --kd ../runs_p6/e_lorma_ht4k --kd_T 1.0 --kd_alpha 0.5 --seed 0

# generation: e_full_ht4k/gen_dev_b4_nr4
python gen_summ.py --run ../runs_p6/e_full_ht4k --split dev --tag b4_nr4 --batch 16 --max_docs 100 --beams 4 --temperature 0.0 --top_p 1.0 --rep_pen 1.0 --no_repeat 4 --len_pen 1.0 --min_new 200 --max_new 1024 --len_scale 1.0 --seed 1

# generation: e_full_ht4k/gen_dev_g_nr4
python gen_summ.py --run ../runs_p6/e_full_ht4k --split dev --tag g_nr4 --batch 48 --max_docs 100 --beams 1 --temperature 0.0 --top_p 1.0 --rep_pen 1.0 --no_repeat 4 --len_pen 1.0 --min_new 200 --max_new 1024 --len_scale 1.0 --seed 1

# generation: e_full_ht4k/gen_test_b4_nr4
python gen_summ.py --run ../runs_p6/e_full_ht4k --split test --tag b4_nr4 --batch 16 --beams 4 --temperature 0.0 --top_p 1.0 --rep_pen 1.0 --no_repeat 4 --len_pen 1.0 --min_new 200 --max_new 1024 --len_scale 1.0 --seed 1

# generation: e_houlsby_ht4k/gen_dev_b4_nr4
python gen_summ.py --run ../runs_p6/e_houlsby_ht4k --split dev --tag b4_nr4 --batch 16 --max_docs 100 --beams 4 --temperature 0.0 --top_p 1.0 --rep_pen 1.0 --no_repeat 4 --len_pen 1.0 --min_new 200 --max_new 1024 --len_scale 1.0 --seed 1

# generation: e_houlsby_ht4k/gen_dev_g_nr4
python gen_summ.py --run ../runs_p6/e_houlsby_ht4k --split dev --tag g_nr4 --batch 48 --max_docs 100 --beams 1 --temperature 0.0 --top_p 1.0 --rep_pen 1.0 --no_repeat 4 --len_pen 1.0 --min_new 200 --max_new 1024 --len_scale 1.0 --seed 1

# generation: e_houlsby_ht4k/gen_test_b4_nr4
python gen_summ.py --run ../runs_p6/e_houlsby_ht4k --split test --tag b4_nr4 --batch 16 --beams 4 --temperature 0.0 --top_p 1.0 --rep_pen 1.0 --no_repeat 4 --len_pen 1.0 --min_new 200 --max_new 1024 --len_scale 1.0 --seed 1

# generation: e_lora_ht4k/gen_dev_b4_nr4
python gen_summ.py --run ../runs_p6/e_lora_ht4k --split dev --tag b4_nr4 --batch 16 --max_docs 100 --beams 4 --temperature 0.0 --top_p 1.0 --rep_pen 1.0 --no_repeat 4 --len_pen 1.0 --min_new 200 --max_new 1024 --len_scale 1.0 --seed 1

# generation: e_lora_ht4k/gen_dev_g_nr4
python gen_summ.py --run ../runs_p6/e_lora_ht4k --split dev --tag g_nr4 --batch 48 --max_docs 100 --beams 1 --temperature 0.0 --top_p 1.0 --rep_pen 1.0 --no_repeat 4 --len_pen 1.0 --min_new 200 --max_new 1024 --len_scale 1.0 --seed 1

# generation: e_lora_ht4k/gen_test_b4_nr4
python gen_summ.py --run ../runs_p6/e_lora_ht4k --split test --tag b4_nr4 --batch 16 --beams 4 --temperature 0.0 --top_p 1.0 --rep_pen 1.0 --no_repeat 4 --len_pen 1.0 --min_new 200 --max_new 1024 --len_scale 1.0 --seed 1

# generation: e_lorma_ht4k/gen_dev_b4_nr4
python gen_summ.py --run ../runs_p6/e_lorma_ht4k --split dev --tag b4_nr4 --batch 16 --max_docs 100 --beams 4 --temperature 0.0 --top_p 1.0 --rep_pen 1.0 --no_repeat 4 --len_pen 1.0 --min_new 200 --max_new 1024 --len_scale 1.0 --seed 1

# generation: e_lorma_ht4k/gen_dev_g_nr4
python gen_summ.py --run ../runs_p6/e_lorma_ht4k --split dev --tag g_nr4 --batch 48 --max_docs 100 --beams 1 --temperature 0.0 --top_p 1.0 --rep_pen 1.0 --no_repeat 4 --len_pen 1.0 --min_new 200 --max_new 1024 --len_scale 1.0 --seed 1

# generation: e_lorma_ht4k/gen_dev_g_nr4_ls13
python gen_summ.py --run ../runs_p6/e_lorma_ht4k --split dev --tag g_nr4_ls13 --batch 48 --max_docs 100 --beams 1 --temperature 0.0 --top_p 1.0 --rep_pen 1.0 --no_repeat 4 --len_pen 1.0 --min_new 200 --max_new 1024 --len_scale 1.3 --seed 1

# generation: e_lorma_ht4k/gen_dev_g_nr6
python gen_summ.py --run ../runs_p6/e_lorma_ht4k --split dev --tag g_nr6 --batch 48 --max_docs 100 --beams 1 --temperature 0.0 --top_p 1.0 --rep_pen 1.0 --no_repeat 6 --len_pen 1.0 --min_new 200 --max_new 1024 --len_scale 1.0 --seed 1

# generation: e_lorma_ht4k/gen_test_b4_nr4
python gen_summ.py --run ../runs_p6/e_lorma_ht4k --split test --tag b4_nr4 --batch 16 --beams 4 --temperature 0.0 --top_p 1.0 --rep_pen 1.0 --no_repeat 4 --len_pen 1.0 --min_new 200 --max_new 1024 --len_scale 1.0 --seed 1

# generation: f_lorma_ht4k/gen_dev_b4_nr4
python gen_summ.py --run ../runs_p6/f_lorma_ht4k --split dev --tag b4_nr4 --batch 16 --max_docs 100 --beams 4 --temperature 0.0 --top_p 1.0 --rep_pen 1.0 --no_repeat 4 --len_pen 1.0 --min_new 200 --max_new 1024 --len_scale 1.0 --seed 1

# generation: f_lorma_ht4k/gen_dev_g_nr4
python gen_summ.py --run ../runs_p6/f_lorma_ht4k --split dev --tag g_nr4 --batch 48 --max_docs 100 --beams 1 --temperature 0.0 --top_p 1.0 --rep_pen 1.0 --no_repeat 4 --len_pen 1.0 --min_new 200 --max_new 1024 --len_scale 1.0 --seed 1

# generation: f_lorma_ht4k/gen_test_b4_nr4
python gen_summ.py --run ../runs_p6/f_lorma_ht4k --split test --tag b4_nr4 --batch 16 --beams 4 --temperature 0.0 --top_p 1.0 --rep_pen 1.0 --no_repeat 4 --len_pen 1.0 --min_new 200 --max_new 1024 --len_scale 1.0 --seed 1

# generation: kd_e10_from_lorma/gen_dev_b4_nr4
python gen_summ.py --run ../runs_p6/kd_e10_from_lorma --split dev --tag b4_nr4 --batch 16 --max_docs 100 --beams 4 --temperature 0.0 --top_p 1.0 --rep_pen 1.0 --no_repeat 4 --len_pen 1.0 --min_new 200 --max_new 1024 --len_scale 1.0 --seed 1

# generation: kd_e10_from_lorma/gen_dev_g_nr4
python gen_summ.py --run ../runs_p6/kd_e10_from_lorma --split dev --tag g_nr4 --batch 48 --max_docs 100 --beams 1 --temperature 0.0 --top_p 1.0 --rep_pen 1.0 --no_repeat 4 --len_pen 1.0 --min_new 200 --max_new 1024 --len_scale 1.0 --seed 1

# generation: kd_e10_from_lorma/gen_test_b4_nr4
python gen_summ.py --run ../runs_p6/kd_e10_from_lorma --split test --tag b4_nr4 --batch 16 --beams 4 --temperature 0.0 --top_p 1.0 --rep_pen 1.0 --no_repeat 4 --len_pen 1.0 --min_new 200 --max_new 1024 --len_scale 1.0 --seed 1
