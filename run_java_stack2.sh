#!/bin/bash

DATA_PREFIX=ast_cfc
NUM_STRUCTURE_TOKENS=10

TRAIN_DATADIR=./stack2_java_test_1k
VALID_DATADIR=./stack2_java_test_1k

TEXT_MODEL_ID=Qwen/Qwen2.5-Coder-7B

STRUCTURE_MODEL_ID=microsoft/unixcoder-base

PROJECTOR_CHECKPOIN=./ckpt/qwen7_unixcoder_3l_java_2emes_stack2.ckpt

TRAINING_STAGE=1
NUM_DEVICES=2
CUDA_LAUNCH_BLOCKING=1
TORCH_USE_CUDA_DSA=1 
python test_java.py \
    --num_workers 96 \
    --devices $NUM_DEVICES \
    --num_nodes 1 \
    --accelerator gpu \
    --text_model_id $TEXT_MODEL_ID \
    --structure_model_id $STRUCTURE_MODEL_ID \
    --projector_checkpoint $PROJECTOR_CHECKPOINT \
    --projector '3L' \
    --language java \
    --dropout_p 0. \
    --default_root_dir ./ \
    --data_prefix $DATA_PREFIX \
    --train_datadir $TRAIN_DATADIR \
    --valid_datadir $VALID_DATADIR \
    --log_dir ./logs/ \
    --seed 1234 \
    --lr 1e-5 \
    --lr_scheduler_type None \
    --weight_decay 0. \
    --gradient_clip_val 1.0 \
    --max_steps -1 \
    --max_epochs 1 \
    --warmup_steps 100 \
    --train_batch_size $NUM_DEVICES \
    --valid_batch_size $NUM_DEVICES \
    --accumulate_grad_batches 32 \
    --training_stage $TRAINING_STAGE \
    --loss mle \
    --alpha_ce .5 \
    --alpha_scst .5 \
    --log_every_n_steps 10 \
    --save_step_frequency 10000000 \
    --val_check_interval 10 \
    --num_structure_tokens ${NUM_STRUCTURE_TOKENS} \
    --precision 'bf16-true' \
    --alpha_kl .0 \
    --kl_temperature 1.0 \
    --reward '' \
    --alpha_align 0.0 
