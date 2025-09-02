#!/bin/bash

DATA_PREFIX=code_cfc_jina
NUM_STRUCTURE_TOKENS=10
EXPERIMENT=bm25_cp_final_copy
TRAINING_STAGE=1

# TEXT_MODEL_ID=bigcode/starcoderbase-1b
TEXT_MODEL_ID=Qwen/Qwen2.5-Coder-1.5B
# TEXT_MODEL_ID=Qwen/Qwen2.5-Coder-1.5B-Instruct
# STRUCTURE_MODEL_ID=microsoft/unixcoder-base
STRUCTURE_MODEL_ID=jinaai/jina-embeddings-v2-base-en
# STRUCTURE_MODEL_ID=microsoft/graphcodebert-base

# MODEL_CHECKPOINT=$(ls -t "lightning_logs/qwen_stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}/checkpoints" | head -n 1)
# MODEL_CHECKPOINT=lightning_logs/qwen_stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}/checkpoints/${MODEL_CHECKPOINT}
MODEL_CHECKPOINT=/home/jovyan/shares/SR006.nfs2/sukhorukov/LlavaCode/lightning_logs/qwen_stage=1_code_cfc_jina_flamingo_final_copy/checkpoints/epoch=0-step=500.ckpt
echo $MODEL_CHECKPOINT
OUTPUT_DIR=results_qwen/stage=${TRAINING_STAGE}_${DATA_PREFIX}_${EXPERIMENT}_pretrained
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=7 python run_benchmark.py \
    --prompt_file /home/jovyan/shares/SR006.nfs2/sukhorukov/LlavaCode/benchmarks_new_vanilla/python_line_completion_sparse_bm25_chunk_10_query_10_window_5.jsonl \
    --text_model_id $TEXT_MODEL_ID \
    --structure_model_id $STRUCTURE_MODEL_ID \
    --task line_completion \
    --compute_cceval_metric 0 \
    --data_prefix $DATA_PREFIX \
    --num_structure_tokens $NUM_STRUCTURE_TOKENS \
    --output_dir $OUTPUT_DIR \
    --language python \
    --model_checkpoint $MODEL_CHECKPOINT \

# --prompt_file /home/jovyan/sukhorukov/codegen/repo_eval/processed_data/python_line_completion_sparse_rg1.jsonl
# python_line_completion_sparse_rg1_uxcpool.jsonl
# python_line_completion_sparse_rg1_uxccls.jsonl
 # python_line_completion_sparse_rg1_uxcpoolast.jsonl
 # python_line_completion_sparse_rg1_jina.jsonl