#!/bin/bash

DATA_PREFIX=code_cfc
NUM_STRUCTURE_TOKENS=10
EXPERIMENT=1500k_kl
TRAINING_STAGE=1

# TEXT_MODEL_ID=bigcode/starcoderbase-1b
TEXT_MODEL_ID=Qwen/Qwen2.5-Coder-1.5B
# STRUCTURE_MODEL_ID=microsoft/unixcoder-base
STRUCTURE_MODEL_ID=jinaai/jina-embeddings-v2-base-en
# STRUCTURE_MODEL_ID=microsoft/graphcodebert-base

# MODEL_CHECKPOINT=$(ls -t "lightning_logs/qwen_stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}/checkpoints" | head -n 1)
# MODEL_CHECKPOINT=lightning_logs/qwen_stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}/checkpoints/${MODEL_CHECKPOINT}
# MODEL_CHECKPOINT=lightning_logs/qwen_stage=${TRAINING_STAGE}_${DATA_PREFIX}_750k/checkpoints/epoch=1-step=20568.ckpt
MODEL_CHECKPOINT=lightning_logs/qwen_stage=1_code_cfc_1500k_kl/checkpoints/epoch=0-step=13800.ckpt
OUTPUT_DIR=results_qwen/stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}
# OUTPUT_DIR=results_qwen/untrained_${DATA_PREFIX}_${NUM_STRUCTURE_TOKENS}t
mkdir -p ${OUTPUT_DIR}

python run_benchmark.py \
    --prompt_file /home/jovyan/sukhorukov/codegen/repo_eval/processed_data/python_line_completion_sparse_rg1.jsonl \
    --text_model_id $TEXT_MODEL_ID \
    --structure_model_id $STRUCTURE_MODEL_ID \
    --task line_completion \
    --compute_cceval_metric 0 \
    --data_prefix $DATA_PREFIX \
    --num_structure_tokens $NUM_STRUCTURE_TOKENS \
    --model_checkpoint $MODEL_CHECKPOINT \
    --output_dir $OUTPUT_DIR \
    --language python

# --prompt_file /home/jovyan/sukhorukov/codegen/repo_eval/processed_data/python_line_completion_sparse_rg1.jsonl
# python_line_completion_sparse_rg1_uxcpool.jsonl
# python_line_completion_sparse_rg1_uxccls.jsonl
 # python_line_completion_sparse_rg1_uxcpoolast.jsonl
 # python_line_completion_sparse_rg1_jina.jsonl