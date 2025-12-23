#!/bin/bash

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_prefix)
      DATA_PREFIX="$2"
      shift 2
      ;;
    --num_structure_tokens)
      NUM_STRUCTURE_TOKENS="$2"
      shift 2
      ;;
    --experiment)
      EXPERIMENT="$2"
      shift 2
      ;;
    --num_beams)
      NUM_BEAMS="$2"
      shift 2
      ;;
    --gen_length)
      GEN_LENGTH="$2"
      shift 2
      ;;
    --num_return_sequences)
      NUM_RETURN_SEQUENCES="$2"
      shift 2
      ;;
    --text_model_id)
      TEXT_MODEL_ID="$2"
      shift 2
      ;;
    --structure_model_id)
      STRUCTURE_MODEL_ID="$2"
      shift 2
      ;;
    --benchmark)
      BENCHMARK="$2"
      shift 2
      ;;
    --model_checkpoint)
      MODEL_CHECKPOINT="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --compute_cceval_metric)
      COMPUTE_CCEVAL_METRIC="$2"
      shift 2
      ;;
    --cfc_place)
      CFC_PLACE="$2"
      shift 2
      ;;
    --projector)
      PROJECTOR="$2"
      shift 2
      ;;
    --language)
      LANGUAGE="$2"
      shift 2
      ;;
    --do_sample)
      DO_SAMPLE="--do_sample"
      shift
      ;;
    --base_dir)
      BASE_DIR="$2"
      shift 2
      ;;
    --repos_to_keep)
      shift  # Remove the flag itself
      # Collect all following arguments until we hit another flag
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        REPOS_TO_KEEP+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set defaults if not provided
DATA_PREFIX=${DATA_PREFIX:-ast_cfc}
NUM_STRUCTURE_TOKENS=${NUM_STRUCTURE_TOKENS:-10}
EXPERIMENT=${EXPERIMENT:-repoeval_python_function}
NUM_BEAMS=${NUM_BEAMS:-4}
GEN_LENGTH=${GEN_LENGTH:-300}
NUM_RETURN_SEQUENCES=${NUM_RETURN_SEQUENCES:-10}
TEXT_MODEL_ID=${TEXT_MODEL_ID:-Qwen/Qwen2.5-Coder-7B}
STRUCTURE_MODEL_ID=${STRUCTURE_MODEL_ID:-microsoft/unixcoder-base}
BENCHMARK=${BENCHMARK:-/home/jovyan/sukhorukov/LlavaCode/python_function_completion_sparse_rg1_bm25_chunk20_window10_query20_top10.jsonl}
MODEL_CHECKPOINT=${MODEL_CHECKPOINT:-/home/jovyan/cherniuk/LlavaCode/ckpt/qwen7_unixcoder_3l_python_emes_stack.ckpt}
TASK=${TASK:-function_completion}
COMPUTE_CCEVAL_METRIC=${COMPUTE_CCEVAL_METRIC:-0}
CFC_PLACE=${CFC_PLACE:-preprefix}
PROJECTOR=${PROJECTOR:-3L}
LANGUAGE=${LANGUAGE:-python}
BASE_DIR=${BASE_DIR:-/home/jovyan/sukhorukov/LlavaCode}
OUTPUT_DIR=${OUTPUT_DIR:-results_python_test/qwen7_${DATA_PREFIX}_${EXPERIMENT}_beam_${NUM_BEAMS}_seq_${NUM_RETURN_SEQUENCES}_gen_${GEN_LENGTH}}
mkdir -p ${OUTPUT_DIR}

if [ ${#REPOS_TO_KEEP[@]} -eq 0 ]; then
  REPOS_TO_KEEP=('amazon-science_patchcore-inspection' 'deepmind_tracr' 'facebookresearch_omnivore' 'leopard-ai_betty' 'maxhumber_redframes')
fi

python run_benchmark_pass.py \
    --prompt_file $BENCHMARK \
    --text_model_id $TEXT_MODEL_ID \
    --structure_model_id $STRUCTURE_MODEL_ID \
    --task function_completion \
    --compute_cceval_metric 0 \
    --data_prefix $DATA_PREFIX \
    --num_structure_tokens $NUM_STRUCTURE_TOKENS \
    --output_dir $OUTPUT_DIR \
    --cfc_place preprefix \
    --model_checkpoint $MODEL_CHECKPOINT \
    --projector '3L' \
    --num_return_sequences $NUM_RETURN_SEQUENCES \
    --num_beams $NUM_BEAMS \
    --gen_length $GEN_LENGTH \
    --language python \
    --do_sample \
    --base_dir $BASE_DIR \
    --repos_to_keep "${REPOS_TO_KEEP[@]}"