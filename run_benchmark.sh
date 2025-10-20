#!/bin/bash

DATA_PREFIX=default
NUM_STRUCTURE_TOKENS=10
EXPERIMENT=truncated
TRAINING_STAGE=untrained

# TEXT_MODEL_ID=bigcode/starcoderbase-1b
TEXT_MODEL_ID=Qwen/Qwen2.5-Coder-1.5B
# TEXT_MODEL_ID=Qwen/Qwen2.5-Coder-7B
# TEXT_MODEL_ID=Qwen/Qwen2.5-Coder-1.5B-Instruct
# STRUCTURE_MODEL_ID=microsoft/unixcoder-base
# STRUCTURE_MODEL_ID=jinaai/jina-embeddings-v2-base-en
STRUCTURE_MODEL_ID=Qwen/Qwen3-Embedding-0.6B
# STRUCTURE_MODEL_ID=microsoft/graphcodebert-base

OUTPUT_DIR=results/${DATA_PREFIX}_${EXPERIMENT}
mkdir -p ${OUTPUT_DIR}

python run_benchmark.py \
    --prompt_file python_line_completion_from_validation_truncated.jsonl \
    --text_model_id $TEXT_MODEL_ID \
    --structure_model_id $STRUCTURE_MODEL_ID \
    --task line_completion \
    --compute_cceval_metric 0 \
    --data_prefix $DATA_PREFIX \
    --num_structure_tokens $NUM_STRUCTURE_TOKENS \
    --output_dir $OUTPUT_DIR \
    --language python
# --model_checkpoint $MODEL_CHECKPOINT \
