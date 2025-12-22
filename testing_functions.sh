#!/bin/bash

# Exit on any error
set -e

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --base_dir)
      BASE_DIR="$2"
      shift 2
      ;;
    --python_version)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --repo_path)
      REPO_PATH="$2"
      shift 2
      ;;
    --results_folder_name)
      RESULTS_FOLDER_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set defaults if not provided
BASE_DIR=${BASE_DIR:-/home/jovyan/sukhorukov/LlavaCode}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
RESULTS_FOLDER_NAME=${RESULTS_FOLDER_NAME:-results_vanilla_functions_report_final}



# Extract filename using parameter expansion
file_name="${REPO_PATH##*/}"
echo $REPO_PATH


ENV_NAME="test_env_$file_name"
REPO_DIR="${BASE_DIR}/repositories/$REPO_PATH"    # Or specify path to repo

mkdir -p ${BASE_DIR}/${RESULTS_FOLDER_NAME}/
TEST_OUTPUT_FILE="${BASE_DIR}/${RESULTS_FOLDER_NAME}/${file_name}.out"


# 1. Create and activate conda environment
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# pip install -U pip

# 2. Install dependencies
# if [ -f "$REPO_DIR/requirements/requirements.txt" ]; then
#     pip install -r "$REPO_DIR/requirements/requirements.txt"
# fi

if [ -f "$REPO_DIR/setup.py" ]; then
    pip install -e "$REPO_DIR"
fi


if [[ "$REPO_DIR" == *"omnivore"* || "$REPO_DIR" == *"redframes"* || "$REPO_DIR" == *"CarperAI_trlx"* ]]; then
    TEST_DIR="$REPO_DIR/tests"
elif [[ "$REPO_DIR" == *"patchcore-inspection"* || "$REPO_DIR" == *"leopard-ai_betty"* ]]; then
    TEST_DIR="$REPO_DIR/test"
elif [[ "$REPO_DIR" == *"imagen-pytorch"* ]]; then
    TEST_DIR="$REPO_DIR/imagen_pytorch/test"
elif [[ "$REPO_DIR" == *"lightweight_mmm"* ]]; then
    TEST_DIR="$REPO_DIR/lightweight_mmm"
    python -m pip install -r "$REPO_DIR/requirements/requirements_tests.txt"
elif [[ "$REPO_DIR" == *"tracr"* ]]; then
    TEST_DIR="$REPO_DIR/tracr/compiler"
fi

# 3. Run tests
python -m pip install pytest

pytest --color=no -v $TEST_DIR 2>&1 | tee $TEST_OUTPUT_FILE
# pytest -v $TEST_DIR | tee "$TEST_OUTPUT_FILE"

# # 4. Delete environment
conda deactivate
# conda remove -y -n "$ENV_NAME" --all
