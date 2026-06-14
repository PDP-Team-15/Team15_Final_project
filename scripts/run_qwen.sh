#!/bin/bash

python llm_evaluation/qwen_inference_2.py \
    --num_shots 0 \
    --temp $1 \
    --top_p $2 \
    --max_token $3 \
    --port $4 \
    --dataset_name qwen2.5_coder_14b_test \
    --prompt_style ${5:-baseline} \
    --model_name ${6:-Qwen/Qwen2.5-Coder-7B-Instruct}

# Example usage:
# 7b 
# bash scripts/run_qwen.sh 0.2 0.9 2048 8003 strict

# 14b
# bash scripts/run_qwen.sh 0.2 0.9 4096 8004 strict Qwen/Qwen2.5-Coder-14B-Instruct-AWQ