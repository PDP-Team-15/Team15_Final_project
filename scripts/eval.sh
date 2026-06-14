
# llama-8b
# --run_names "llama_8b_test_shot_0_m_2048_t_0.2_p_0.9"
# qwen-7b
# --run_names "qwen_7b_test_shot_0_m_2048_t_0.2_p_0.9"

# python eval/run_and_speedup/eval_run_new.py \
#     --target cuda omp \
#     --from_apis omp cuda \
#     --run_names "qwen2.5_coder_14b_test_user_prompt_strict_shot_0_m_4096_t_0.2_p_0.9" \
#     --log Datasets/eval/qwen2.5_coder_14b_test_user_prompt_strict_shot_0_m_4096_t_0.2_p_0.9/eval_run.log

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate unipar
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH  

RUN_NAME="qwen2.5_coder_14b_test_baseline_shot_0_m_4096_t_0.2_p_0.9"

    python eval/run_and_speedup/eval_run_new.py \
    --target cuda omp \
    --from_apis omp cuda \
    --run_names "$RUN_NAME" \
    --log "Datasets/eval/$RUN_NAME/eval_run.log"

    