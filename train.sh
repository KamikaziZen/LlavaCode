#!/bin/bash

DATA_PREFIX=ast_cfc
NUM_STRUCTURE_TOKENS=10
EXPERIMENT=random

TRAIN_DATADIR=/home/jovyan/sukhorukov/codegen/data_python_20_lines_15_top_1_linecompletion/prepared_data/processed/train
VALID_DATADIR=/home/jovyan/sukhorukov/codegen/data_python_20_lines_15_top_1_linecompletion/prepared_data/processed/valid
TEXT_MODEL_ID=bigcode/starcoderbase-1b
STRUCTURE_MODEL_ID=microsoft/unixcoder-base

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
    --exp_name stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}

MODEL_CHECKPOINT=$(ls -t "lightning_logs/stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}/checkpoints" | head -n 1)
MODEL_CHECKPOINT=lightning_logs/stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}/checkpoints/${MODEL_CHECKPOINT}
OUTPUT_DIR=results/stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}
mkdir -p ${OUTPUT_DIR}

python run_benchmark.py \
    --prompt_file /home/jovyan/sukhorukov/codegen/repo_eval/processed_data/python_line_completion_sparse_rg1.jsonl \
    --text_model_id $TEXT_MODEL_ID \
    --structure_model_id $STRUCTURE_MODEL_ID \
    --task line_completion \
    --compute_cceval_metric 0 \
    --data_prefix $DATA_PREFIX \
    --num_structure_tokens ${NUM_STRUCTURE_TOKENS} \
    --output_dir $OUTPUT_DIR \
    --model_checkpoint $MODEL_CHECKPOINT \
    --language python

TRAINING_STAGE=1
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
    --lr 2e-4 \
    --lr_scheduler_type cosine \
    --weight_decay 0. \
    --gradient_clip_val 1.0 \
    --max_steps -1 \
    --max_epochs 3 \
    --warmup_steps 100 \
    --train_batch_size 16 \
    --valid_batch_size 16 \
    --accumulate_grad_batches 4 \
    --training_stage $TRAINING_STAGE \
    --log_every_n_steps 50 \
    --save_step_frequency 500 \
    --val_check_interval 100 \
    --num_structure_tokens ${NUM_STRUCTURE_TOKENS} \
    --precision 'bf16-mixed' \
    --model_checkpoint $MODEL_CHECKPOINT \
    --exp_name stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}

MODEL_CHECKPOINT=$(ls -t "lightning_logs/stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}/checkpoints" | head -n 1)
MODEL_CHECKPOINT=lightning_logs/stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}/checkpoints/${MODEL_CHECKPOINT}
OUTPUT_DIR=results/stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}
mkdir -p ${OUTPUT_DIR}

python run_benchmark.py \
    --prompt_file /home/jovyan/sukhorukov/codegen/repo_eval/processed_data/python_line_completion_sparse_rg1.jsonl \
    --text_model_id $TEXT_MODEL_ID \
    --structure_model_id $STRUCTURE_MODEL_ID \
    --task line_completion \
    --compute_cceval_metric 0 \
    --data_prefix $DATA_PREFIX \
    --num_structure_tokens ${NUM_STRUCTURE_TOKENS} \
    --output_dir $OUTPUT_DIR \
    --model_checkpoint $MODEL_CHECKPOINT \
    --language python

TRAINING_STAGE=2
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --num_workers 96 \
    --devices 2 \
    --num_nodes 1 \
    --accelerator gpu \
    --text_model_id $TRAIN_DATADIR \
    --structure_model_id $STRUCTURE_MODEL_ID \
    --pad_token_id 0 \
    --dropout_p 0. \
    --default_root_dir ./ \
    --data_prefix $DATA_PREFIX \
    --train_datadir $TRAIN_DATADIR \
    --valid_datadir $VALID_DATADIR \
    --log_dir ./logs/ \
    --seed 1234 \
    --lr 2e-5 \
    --lr_scheduler_type cosine \
    --weight_decay 0. \
    --gradient_clip_val 1.0 \
    --max_steps -1 \
    --max_epochs 1 \
    --warmup_steps 50 \
    --train_batch_size 16 \
    --valid_batch_size 16 \
    --accumulate_grad_batches 4 \
    --training_stage $TRAINING_STAGE \
    --log_every_n_steps 50 \
    --save_step_frequency 500 \
    --val_check_interval 100 \
    --num_structure_tokens ${NUM_STRUCTURE_TOKENS} \
    --model_checkpoint $MODEL_CHECKPOINT \
    --precision 'bf16-mixed' \
    --exp_name stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}

MODEL_CHECKPOINT=$(ls -t "lightning_logs/stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}/checkpoints" | head -n 1)
MODEL_CHECKPOINT=lightning_logs/stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}/checkpoints/${MODEL_CHECKPOINT}
OUTPUT_DIR=results/stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}
mkdir -p ${OUTPUT_DIR}

python run_benchmark.py \
    --prompt_file /home/jovyan/sukhorukov/codegen/repo_eval/processed_data/python_line_completion_sparse_rg1.jsonl \
    --text_model_id $TEXT_MODEL_ID \
    --structure_model_id $STRUCTURE_MODEL_ID \
    --task line_completion \
    --compute_cceval_metric 0 \
    --data_prefix $DATA_PREFIX \
    --num_structure_tokens ${NUM_STRUCTURE_TOKENS} \
    --output_dir $OUTPUT_DIR \
    --model_checkpoint $MODEL_CHECKPOINT \
    --language python