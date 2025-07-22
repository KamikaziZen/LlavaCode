#!/bin/bash

python memorycell.py \
    --model_name bigcode/starcoderbase-1b \
    --dtype float32 \
    --N_mem_tokens 1 \
    --max_length 512 \
    --num_iterations 5000 \
    --lr 1e-2 \
    --weight_decay 0.01 \
    --start_index 0 \
    --end_index 5 \
    --texts_path /home/jovyan/sukhorukov/codegen/data_python_20_lines_15_top_1_linecompletion/prepared_data/processed/train
