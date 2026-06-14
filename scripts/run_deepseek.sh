#!/bin/bash

python llm_evaluation/deepseek_inference.py \
    --num_shots ${6:-0} \
    --temp $1 \
    --top_p $2 \
    --max_token $3 \
    --port $4 \
    --dataset_name deepseek_32b_strict_v3_cuda_omp \
    --prompt_style ${5:-baseline} \
    --direction ${7:-both}

# Example usage:
# cuda->omp only: bash scripts/run_deepseek.sh 0.2 0.9 11000 8006 strict 0 cuda_omp
# omp->cuda only: bash scripts/run_deepseek.sh 0.2 0.9 11000 8006 strict 0 omp_cuda
# both:           bash scripts/run_deepseek.sh 0.2 0.9 11000 8006 strict
