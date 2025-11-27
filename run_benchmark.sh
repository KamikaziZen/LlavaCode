#!/bin/bash

DATA_PREFIX=ast_cfc
NUM_STRUCTURE_TOKENS=10
EXPERIMENT=repoeval

# TEXT_MODEL_ID=Qwen/Qwen2.5-Coder-1.5B
TEXT_MODEL_ID=Qwen/Qwen2.5-Coder-7B
STRUCTURE_MODEL_ID=microsoft/unixcoder-base
# STRUCTURE_MODEL_ID=Qwen/Qwen3-Embedding-0.6B

BENCHMARK=python_line_completion_sparse_bm25.jsonl

MODEL_CHECKPOINT=/home/jovyan/cherniuk/LlavaCode/ckpt/qwen7_unixcoder_3l_python_emes_stack.ckpt

OUTPUT_DIR=results/qwen7_${DATA_PREFIX}_${EXPERIMENT}
mkdir -p ${OUTPUT_DIR}

python run_benchmark.py \
    --prompt_file $BENCHMARK \
    --text_model_id $TEXT_MODEL_ID \
    --structure_model_id $STRUCTURE_MODEL_ID \
    --task line_completion \
    --compute_cceval_metric 0 \
    --data_prefix $DATA_PREFIX \
    --num_structure_tokens $NUM_STRUCTURE_TOKENS \
    --output_dir $OUTPUT_DIR \
    --cfc_place preprefix \
    --model_checkpoint $MODEL_CHECKPOINT \
    --projector '3L' \
    --gen_length 50 \
    --language python