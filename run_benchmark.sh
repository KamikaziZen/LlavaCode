#!/bin/bash

python run_benchmark.py \
    --task line_completion \
    --prompt_file /home/jovyan/sukhorukov/codegen/repo_eval/processed_data/python_line_completion_sparse_rg1.jsonl \
    --compute_cceval_metric 0 \
    --output_dir llava_trained_nocode_output \
    --model_checkpoint 'lightning_logs/version_10/checkpoints/epoch=0-step=700.ckpt' \
    --language python

# 'lightning_logs/version_9/checkpoints/epoch=4-step=3776.ckpt'
# 'lightning_logs/version_10/checkpoints/epoch=0-step=700.ckpt'