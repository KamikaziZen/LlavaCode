#!/bin/bash

DATA_PREFIX=ast_cfc
NUM_STRUCTURE_TOKENS=10

TEXT_MODEL_ID=Qwen/Qwen2.5-Coder-7B
STRUCTURE_MODEL_ID=microsoft/unixcoder-base

BENCHMARK=/home/jovyan/gusak/LlavaCode/java_line_completion_sparse_bm25.jsonl
LANGUAGE=java

# MODEL_CHECKPOINT=./ckpt/qwen7_unixcoder_3l_java_2emes_stack2.ckpt
PROJECTOR_CHECKPOINT=./ckpt/qwen7_unixcoder_3l_java_2emes_stack2_projector.pth

OUTPUT_DIR=results/qwen7_${DATA_PREFIX}_${LANGUAGE}
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
    --projector_checkpoint $PROJECTOR_CHECKPOINT \
    --projector '3L' \
    --gen_length 50 \
    --language $LANGUAGE 
