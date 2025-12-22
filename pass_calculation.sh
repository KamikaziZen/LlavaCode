#!/bin/bash

set -e

DATA_PREFIX=ast_cfc
NUM_STRUCTURE_TOKENS=10
EXPERIMENT=repoeval_python_function
NUM_BEAMS=4
GEN_LENGTH=300
NUM_RETURN_SEQUENCES=10
TOPK_PASS=10
TEXT_MODEL_ID=Qwen/Qwen2.5-Coder-7B
STRUCTURE_MODEL_ID=microsoft/unixcoder-base
BENCHMARK=/home/jovyan/sukhorukov/LlavaCode/python_function_completion_sparse_rg1_bm25_chunk20_window10_query20_top10.jsonl
MODEL_CHECKPOINT=/home/jovyan/cherniuk/LlavaCode/ckpt/qwen7_unixcoder_3l_python_emes_stack.ckpt
TASK=function_completion
PROJECTOR=3L
LANGUAGE=python

OUTPUT_DIR=results_python_pass_script_testing/qwen7_${DATA_PREFIX}_${EXPERIMENT}_beam_${NUM_BEAMS}_seq_${NUM_RETURN_SEQUENCES}_gen_${GEN_LENGTH}

BASE_DIR=/home/jovyan/sukhorukov/LlavaCode

REPOS_TO_KEEP=('amazon-science_patchcore-inspection' 'facebookresearch_omnivore' 'leopard-ai_betty' 'maxhumber_redframes')

jsonl_file="$OUTPUT_DIR/input_for_testing_function_${DATA_PREFIX}_report.jsonl"
result_dir="$BASE_DIR/test_results_function_${DATA_PREFIX}_report_final"


bash "${BASE_DIR}/run_benchmark_pass.sh" \
  --data_prefix "$DATA_PREFIX" \
  --num_structure_tokens "$NUM_STRUCTURE_TOKENS" \
  --experiment "$EXPERIMENT" \
  --num_beams "$NUM_BEAMS" \
  --gen_length "$GEN_LENGTH" \
  --num_return_sequences "$NUM_RETURN_SEQUENCES" \
  --text_model_id "$TEXT_MODEL_ID" \
  --structure_model_id "$STRUCTURE_MODEL_ID" \
  --benchmark "$BENCHMARK" \
  --model_checkpoint "$MODEL_CHECKPOINT" \
  --task "$TASK" \
  --projector "$PROJECTOR" \
  --language "$LANGUAGE" \
  --output_dir "$OUTPUT_DIR" \
  --base_dir "$BASE_DIR" \
  --repos_to_keep "${REPOS_TO_KEEP[@]}"

python "${BASE_DIR}/create_envs_run_vanilla_tests.py" \
  --base_dir "$BASE_DIR" \
  --repos "${REPOS_TO_KEEP[@]}" \
  --results_folder_name "results_vanilla_functions_report_final"


python "${BASE_DIR}/run_tests_function.py" \
  --jsonl_file "$jsonl_file" \
  --base_dir "$BASE_DIR" \
  --result_dir "$result_dir" \
  --repos "${REPOS_TO_KEEP[@]}"

python "${BASE_DIR}/calculate_pass_metric.py" \
  --path "$result_dir" \
  --vanilla_path "$BASE_DIR/results_vanilla_functions_report_final" \
  --prefixes "$DATA_PREFIX" \
  --k $TOPK_PASS \
  --repos "${REPOS_TO_KEEP[@]}"