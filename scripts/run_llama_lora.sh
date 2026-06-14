#!/bin/bash

python llm_evaluation/llama_inference_2_lora.py \
    --num_shots ${6:-0} \
    --temp $1 \
    --top_p $2 \
    --max_token $3 \
    --port $4 \
    --dataset_name llama3.1_8b_eval \
    --prompt_style ${5:-baseline}

# Example usage:
# zero-shot: bash scripts/run_llama.sh 0.2 0.9 2048 8002 strict
# few-shot:  bash scripts/run_llama.sh 0.2 0.9 2048 8002 strict 1