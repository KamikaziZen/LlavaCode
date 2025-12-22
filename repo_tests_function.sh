#!/bin/bash
REPO="$1"
FILEPATH="$2"
TMP_CONTENT="$3"
EXP_NAME="$4"
RESULTS_DIR="$5"
BASE_DIR="$6"
REPO_DIR="${BASE_DIR}/repositories/${REPO}"
mkdir -p "$RESULTS_DIR"

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
ENV_NAME="test_env_$REPO"
conda activate "$ENV_NAME"


# Save original content
ORIGCONTENT=$(cat "$FILEPATH")

# Overwrite file
cp "$TMP_CONTENT" "$FILEPATH"


# REPO_DIR="/home/jovyan/sukhorukov/LlavaCode/repositories/$1"

if [[ "$REPO_DIR" == *"omnivore"* || "$REPO_DIR" == *"redframes"* || "$REPO_DIR" == *"CarperAI_trlx"* ]]; then
    TEST_DIR="$REPO_DIR/tests"
elif [[ "$REPO_DIR" == *"patchcore-inspection"* || "$REPO_DIR" == *"leopard-ai_betty"* ]]; then
    TEST_DIR="$REPO_DIR/test"
elif [[ "$REPO_DIR" == *"imagen-pytorch"* ]]; then
    TEST_DIR="$REPO_DIR/imagen_pytorch/test"
elif [[ "$REPO_DIR" == *"lightweight_mmm"* ]]; then
    TEST_DIR="$REPO_DIR/lightweight_mmm"
    # python -m pip install -r "$REPO_DIR/requirements/requirements_tests.txt"
elif [[ "$REPO_DIR" == *"tracr"* ]]; then
    TEST_DIR="$REPO_DIR/tracr/compiler"
fi

pytest --color=no -v $TEST_DIR 2>&1 | tee "$RESULTS_DIR/${REPO}_${EXP_NAME}.out"
# pytest -v $TEST_DIR | tee "$TEST_OUTPUT_FILE"
# pytest > "../$RESULTS_DIR/${REPO}_$(basename "$FILEPATH").out" 2>&1

# Restore original content
echo "$ORIGCONTENT" > "$FILEPATH"

conda deactivate
