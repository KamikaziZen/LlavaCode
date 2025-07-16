#!/bin/bash

DATA_PREFIX=code_cfc
NUM_STRUCTURE_TOKENS=10

TRAINING_STAGE=0
# CUDA_VISIBLE_DEVICES=0,1 python train.py \
#     --num_workers 96 \
#     --devices 2 \
#     --num_nodes 1 \
#     --accelerator gpu \
#     --text_model_id bigcode/starcoderbase-1b \
#     --structure_model_id microsoft/unixcoder-base \
#     --dropout_p 0. \
#     --default_root_dir ./ \
#     --data_prefix $DATA_PREFIX \
#     --train_datadir '/home/jovyan/sukhorukov/codegen/data_python_10_lines_10_top_1_linecompletion/prepared_data/processed/train' \
#     --valid_datadir '/home/jovyan/sukhorukov/codegen/data_python_10_lines_10_top_1_linecompletion/prepared_data/processed/valid' \
#     --log_dir ./logs/ \
#     --seed 1234 \
#     --lr 2e-3 \
#     --lr_scheduler_type cosine \
#     --weight_decay 0. \
#     --gradient_clip_val 1.0 \
#     --max_steps -1 \
#     --max_epochs 3 \
#     --warmup_steps 200 \
#     --train_batch_size 16 \
#     --valid_batch_size 16 \
#     --accumulate_grad_batches 4 \
#     --training_stage $TRAINING_STAGE \
#     --log_every_n_steps 50 \
#     --save_step_frequency 500 \
#     --val_check_interval 100 \
#     --num_structure_tokens ${NUM_STRUCTURE_TOKENS} \
#     --precision 'bf16-mixed' \
#     --exp_name projection_${DATA_PREFIX}_stage=${TRAINING_STAGE}

TRAINING_STAGE=1
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --num_workers 96 \
    --devices 2 \
    --num_nodes 1 \
    --accelerator gpu \
    --text_model_id bigcode/starcoderbase-1b \
    --structure_model_id jinaai/jina-embeddings-v2-base-en \
    --dropout_p 0. \
    --default_root_dir ./ \
    --data_prefix $DATA_PREFIX \
    --train_datadir '/home/jovyan/sukhorukov/codegen/data_python_10_lines_10_top_1_linecompletion/prepared_data/processed/train' \
    --valid_datadir '/home/jovyan/sukhorukov/codegen/data_python_10_lines_10_top_1_linecompletion/prepared_data/processed/valid' \
    --log_dir ./logs/ \
    --seed 1234 \
    --lr 2e-3 \
    --lr_scheduler_type cosine \
    --weight_decay 0. \
    --gradient_clip_val 1.0 \
    --max_steps -1 \
    --max_epochs 3 \
    --warmup_steps 200 \
    --train_batch_size 16 \
    --valid_batch_size 16 \
    --accumulate_grad_batches 4 \
    --training_stage $TRAINING_STAGE \
    --log_every_n_steps 50 \
    --save_step_frequency 500 \
    --val_check_interval 100 \
    --num_structure_tokens ${NUM_STRUCTURE_TOKENS} \
    --precision 'bf16-mixed' \
    --exp_name ${DATA_PREFIX}_stage=${TRAINING_STAGE}

#     --train_datadir /home/jovyan/sukhorukov/codegen/data_python/prepared_data/processed/train \
# --valid_datadir /home/jovyan/sukhorukov/codegen/data_python/prepared_data/processed/valid \

# OUTPUT_DIR=results/projection_trained_${DATA_PREFIX}_premiddle_random
# mkdir -p ${OUTPUT_DIR}

# CUDA_VISIBLE_DEVICES=0 python run_benchmark.py \
#     --prompt_file /home/jovyan/sukhorukov/codegen/repo_eval/processed_data/python_line_completion_sparse_rg1.jsonl \
#     --text_model_id bigcode/starcoderbase-1b \
#     --structure_model_id microsoft/unixcoder-base \
#     --task line_completion \
#     --compute_cceval_metric 0 \
#     --data_prefix $DATA_PREFIX \
#     --num_structure_tokens ${NUM_STRUCTURE_TOKENS} \
#     --output_dir $OUTPUT_DIR \
#     --model_checkpoint lightning_logs/projection_${DATA_PREFIX}_premiddle_random/checkpoints/epoch=2-step=2238.ckpt \
#     --language python

# TRAINING_STAGE=1
# CUDA_VISIBLE_DEVICES=0,1 python train.py \
#     --num_workers 96 \
#     --devices 2 \
#     --num_nodes 1 \
#     --accelerator gpu \
#     --text_model_id bigcode/starcoderbase-1b \
#     --structure_model_id microsoft/unixcoder-base \
#     --pad_token_id 0 \
#     --dropout_p 0. \
#     --default_root_dir ./ \
#     --data_prefix $DATA_PREFIX \
#     --train_datadir /home/jovyan/sukhorukov/codegen/data_python/prepared_data/processed/train \
#     --valid_datadir /home/jovyan/sukhorukov/codegen/data_python/prepared_data/processed/valid \
#     --log_dir ./logs/ \
#     --seed 1234 \
#     --lr 2e-5 \
#     --lr_scheduler_type cosine \
#     --weight_decay 0. \
#     --gradient_clip_val 1.0 \
#     --max_steps -1 \
#     --max_epochs 1 \
#     --warmup_steps 50 \
#     --train_batch_size 16 \
#     --valid_batch_size 16 \
#     --accumulate_grad_batches 4 \
#     --training_stage $TRAINING_STAGE \
#     --log_every_n_steps 50 \
#     --save_step_frequency 500 \
#     --val_check_interval 100 \
#     --num_structure_tokens ${NUM_STRUCTURE_TOKENS} \
#     --model_checkpoint lightning_logs/projection_${DATA_PREFIX}_premiddle_random/checkpoints/epoch=2-step=2238.ckpt \
#     --precision 'bf16-mixed' \
#     --exp_name finetune_${DATA_PREFIX}_stage=${TRAINING_STAGE}

# OUTPUT_DIR=results/finetune_${DATA_PREFIX}_premiddle_random
# mkdir -p ${OUTPUT_DIR}

# CUDA_VISIBLE_DEVICES=0 python run_benchmark.py \
#     --prompt_file /home/jovyan/sukhorukov/codegen/repo_eval/processed_data/python_line_completion_sparse_rg1.jsonl \
#     --text_model_id bigcode/starcoderbase-1b \
#     --structure_model_id microsoft/unixcoder-base \
#     --task line_completion \
#     --compute_cceval_metric 0 \
#     --data_prefix $DATA_PREFIX \
#     --num_structure_tokens ${NUM_STRUCTURE_TOKENS} \
#     --output_dir $OUTPUT_DIR \
#     --model_checkpoint lightning_logs/finetune_${DATA_PREFIX}_premiddle_random/checkpoints/epoch=0-step=700.ckpt \
#     --language python


# --train_datadir /home/jovyan/sukhorukov/codegen/data_python/prepared_data/processed/train \
# --valid_datadir /home/jovyan/sukhorukov/codegen/data_python/prepared_data/processed/valid \
