#!/bin/bash

DATA_PREFIX=code_cfc
NUM_STRUCTURE_TOKENS=10
# EXPERIMENT="t${NUM_STRUCTURE_TOKENS}"
EXPERIMENT=1500k_uxc_unfreezed

# TRAIN_DATADIR=/home/jovyan/sukhorukov/codegen/data_python_20_lines_15_top_1_linecompletion/prepared_data/processed/train
# VALID_DATADIR=/home/jovyan/sukhorukov/codegen/data_python_20_lines_15_top_1_linecompletion/prepared_data/processed/valid
# TRAIN_DATADIR=/home/jovyan/sukhorukov/codegen/data_python_10_lines_10_top_1_linecompletion/prepared_data/processed/train
# VALID_DATADIR=/home/jovyan/sukhorukov/codegen/data_python_10_lines_10_top_1_linecompletion/prepared_data/processed/valid
TRAIN_DATADIR=/home/jovyan/cherniuk/LlavaCode/processed_1500k/train
VALID_DATADIR=/home/jovyan/cherniuk/LlavaCode/processed_1500k/valid
# TEXT_MODEL_ID=bigcode/starcoderbase-1b
TEXT_MODEL_ID=Qwen/Qwen2.5-Coder-1.5B
# TEXT_MODEL_ID=Qwen/Qwen2.5-Coder-1.5B-Instruct
STRUCTURE_MODEL_ID=microsoft/unixcoder-base
# STRUCTURE_MODEL_ID=jinaai/jina-embeddings-v2-base-en
# STRUCTURE_MODEL_ID=microsoft/graphcodebert-base

# OUTPUT_DIR=results/untrained_${DATA_PREFIX}_${EXPERIMENT}
# mkdir -p ${OUTPUT_DIR}

# CUDA_VISIBLE_DEVICES=0 python run_benchmark.py \
#     --prompt_file /home/jovyan/sukhorukov/codegen/repo_eval/processed_data/python_line_completion_sparse_rg1.jsonl \
#     --text_model_id $TEXT_MODEL_ID \
#     --structure_model_id $STRUCTURE_MODEL_ID \
#     --task line_completion \
#     --compute_cceval_metric 0 \
#     --data_prefix $DATA_PREFIX \
#     --num_structure_tokens ${NUM_STRUCTURE_TOKENS} \
#     --output_dir $OUTPUT_DIR \
#     --language python

TRAINING_STAGE=0
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --num_workers 96 \
    --devices 2 \
    --num_nodes 1 \
    --accelerator gpu \
    --text_model_id $TEXT_MODEL_ID \
    --structure_model_id $STRUCTURE_MODEL_ID \
    --dropout_p 0. \
    --default_root_dir ./ \
    --data_prefix $DATA_PREFIX \
    --train_datadir $TRAIN_DATADIR \
    --valid_datadir $VALID_DATADIR \
    --log_dir ./logs/ \
    --seed 1234 \
    --lr 6e-3 \
    --lr_scheduler_type linear \
    --weight_decay 0. \
    --max_steps -1 \
    --max_epochs 1 \
    --warmup_steps 500 \
    --train_batch_size 12 \
    --valid_batch_size 12 \
    --accumulate_grad_batches 4 \
    --training_stage $TRAINING_STAGE \
    --loss mle \
    --log_every_n_steps 50 \
    --save_step_frequency 500 \
    --val_check_interval 500 \
    --num_structure_tokens 1 \
    --precision 'bf16-mixed' \
    --exp_name qwen_stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}

# MODEL_CHECKPOINT=$(ls -t "lightning_logs/qwen_stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}/checkpoints" | head -n 1)
# MODEL_CHECKPOINT=lightning_logs/qwen_stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}/checkpoints/${MODEL_CHECKPOINT}
# echo $MODEL_CHECKPOINT
# OUTPUT_DIR=results_qwen/stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}
# mkdir -p ${OUTPUT_DIR}

# CUDA_VISIBLE_DEVICES=0 python run_benchmark.py \
#     --prompt_file /home/jovyan/sukhorukov/codegen/repo_eval/processed_data/python_line_completion_sparse_rg1.jsonl \
#     --text_model_id $TEXT_MODEL_ID \
#     --structure_model_id $STRUCTURE_MODEL_ID \
#     --task line_completion \
#     --compute_cceval_metric 0 \
#     --data_prefix $DATA_PREFIX \
#     --num_structure_tokens ${NUM_STRUCTURE_TOKENS} \
#     --output_dir $OUTPUT_DIR \
#     --model_checkpoint $MODEL_CHECKPOINT \
#     --language python

# TRAINING_STAGE=1// The code snippet you provided is a shell script that includes a command to run a
# CUDA_VISIBLE_DEVICES=0,1 python train.py \
#     --num_workers 96 \
#     --devices 2 \
#     --num_nodes 1 \
#     --accelerator gpu \
#     --text_model_id $TEXT_MODEL_ID \
#     --structure_model_id $STRUCTURE_MODEL_ID \
#     --dropout_p 0. \
#     --default_root_dir ./ \
#     --data_prefix $DATA_PREFIX \
#     --train_datadir $TRAIN_DATADIR \
#     --valid_datadir $VALID_DATADIR \
#     --log_dir ./logs/ \
#     --seed 1234 \
#     --lr 2e-3 \
#     --lr_scheduler_type linear \
#     --weight_decay 0. \
#     --gradient_clip_val 1.0 \
#     --max_steps -1 \
#     --max_epochs 3 \
#     --warmup_steps 500 \
#     --train_batch_size 4 \
#     --valid_batch_size 4 \
#     --accumulate_grad_batches 2 \
#     --training_stage $TRAINING_STAGE \
#     --loss mle \
#     --log_every_n_steps 50 \
#     --save_step_frequency 500 \
#     --val_check_interval 600 \
#     --num_structure_tokens ${NUM_STRUCTURE_TOKENS} \
#     --precision 'bf16-mixed' \
#     --alpha_kl 4.0 \
#     --kl_temperature 1.0 \
#     --exp_name qwen_stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}_no0
# --model_checkpoint $MODEL_CHECKPOINT \

# MODEL_CHECKPOINT=$(ls -t "lightning_logs/qwen_stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}/checkpoints" | head -n 1)
# MODEL_CHECKPOINT=lightning_logs/qwen_stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}/checkpoints/${MODEL_CHECKPOINT}
# OUTPUT_DIR=results_qwen/stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}
# mkdir -p ${OUTPUT_DIR}

# CUDA_VISIBLE_DEVICES=0 python run_benchmark.py \
#     --prompt_file /home/jovyan/sukhorukov/codegen/repo_eval/processed_data/python_line_completion_sparse_rg1.jsonl \
#     --text_model_id $TEXT_MODEL_ID \
#     --structure_model_id $STRUCTURE_MODEL_ID \
#     --task line_completion \
#     --compute_cceval_metric 0 \
#     --data_prefix $DATA_PREFIX \
#     --num_structure_tokens $NUM_STRUCTURE_TOKENS \
#     --output_dir $OUTPUT_DIR \
#     --model_checkpoint $MODEL_CHECKPOINT \
#     --language python
    
# TRAINING_STAGE=1
# CUDA_VISIBLE_DEVICES=0,1 python train.py \
#     --num_workers 96 \
#     --devices 2 \
#     --num_nodes 1 \
#     --accelerator gpu \
#     --text_model_id $TEXT_MODEL_ID \
#     --structure_model_id $STRUCTURE_MODEL_ID \
#     --dropout_p 0. \
#     --default_root_dir ./ \
#     --data_prefix $DATA_PREFIX \
#     --train_datadir $TRAIN_DATADIR \
#     --valid_datadir $VALID_DATADIR \
#     --log_dir ./logs/ \
#     --seed 1234 \
#     --lr 2e-5 \
#     --lr_scheduler_type cosine \
#     --weight_decay 0. \
#     --gradient_clip_val 1.0 \
#     --max_steps -1 \
#     --max_epochs 5 \
#     --warmup_steps 50 \
#     --train_batch_size 16 \
#     --valid_batch_size 16 \
#     --accumulate_grad_batches 4 \
#     --training_stage $TRAINING_STAGE \
#     --loss mle \
#     --log_every_n_steps 10 \
#     --save_step_frequency 500 \
#     --val_check_interval 100 \
#     --num_structure_tokens ${NUM_STRUCTURE_TOKENS} \
#     --model_checkpoint $MODEL_CHECKPOINT \
#     --precision 'bf16-mixed' \
#     --exp_name qwen_stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}

# MODEL_CHECKPOINT=$(ls -t "lightning_logs/stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}/checkpoints" | head -n 1)
# MODEL_CHECKPOINT=lightning_logs/stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}/checkpoints/${MODEL_CHECKPOINT}
# OUTPUT_DIR=results/stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}
# mkdir -p ${OUTPUT_DIR}

# CUDA_VISIBLE_DEVICES=0 python run_benchmark.py \
#     --prompt_file /home/jovyan/sukhorukov/codegen/repo_eval/processed_data/python_line_completion_sparse_rg1.jsonl \
#     --text_model_id $TEXT_MODEL_ID \
#     --structure_model_id $STRUCTURE_MODEL_ID \
#     --task line_completion \
#     --compute_cceval_metric 0 \
#     --data_prefix $DATA_PREFIX \
#     --num_structure_tokens ${NUM_STRUCTURE_TOKENS} \
#     --output_dir $OUTPUT_DIR \
#     --model_checkpoint $MODEL_CHECKPOINT \
#     --language python