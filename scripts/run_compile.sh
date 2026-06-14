#!/bin/bash
# Usage: bash scripts/run_compile.sh <model_out_dir> <eval_dir_name>

MODEL_OUT="vllm_llama3.1_8b_test_strict_v1_strict_shots=0_max_token=8192.0_temp=0.2_p=0.9_port=8009"
EVAL_DIR="llama_8b_test_strict_v1_shot_0_m_8192_t_0.2_p_0.9"

python eval/compilation/eval_compile.py \
    --model_out_files "$MODEL_OUT" \
    --extracted_code_paths "$EVAL_DIR"
