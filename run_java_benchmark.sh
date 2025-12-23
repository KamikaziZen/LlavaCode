#!/bin/bash

DATA_PREFIX=ast_cfc
NUM_STRUCTURE_TOKENS=10

TEXT_MODEL_ID=Qwen/Qwen2.5-Coder-7B
STRUCTURE_MODEL_ID=microsoft/unixcoder-base

BENCHMARK=/home/jovyan/gusak/LlavaCode/java_line_completion_sparse_bm25.jsonl
LANGUAGE=java

# PROJECTOR_CHECKPOINT=./ckpt/qwen7_unixcoder_3l_python_emes_stack_projector.pth
PROJECTOR_CHECKPOINT=/mnt/virtual_ai0001053-01336_SR006-nfs3/kamikazi/lightning_logs/qwen7b_ast_cfc_lr5e-5_b64_2*em+es_ent0.9_kl0.0_3L_java_10t/checkpoints/epoch=0-step=1600-Val_Acc_EM=0.7048-Val_Acc_ES=0.8496_projector.pth

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
