#!/bin/bash
# Run compilation + correctness agents on DeepSeek 32B baseline eval dataset
# omp->cuda: 17 kernels (缺 michalewicz — 原始推論未跑到，待補), cuda->omp: 17 kernels (total 34)
# 補完 michalewicz 後需重跑 eval_compile.py，再執行本腳本

DATASET="/home/pdp15/UniPar_AI/Datasets/eval/deepseek_32b_baseline_shot_0_m_8192_t_0.2_p_0.9"
EXTRA_ARGS="$@"   # 例如 --compile_only

cd /home/pdp15/UniPar_AI/multiagent_pipeline

echo "======================================================"
echo " DeepSeek 32B Baseline — Compilation + Run Agent"
echo " Dataset: $DATASET"
echo "======================================================"

echo ""
echo "=== [1/2] omp -> cuda (17 kernels) ==="
conda run -n unipar python datasets_after_eval_compile_main.py \
    --dataset_dir "$DATASET" \
    --from_api omp \
    --to_api cuda \
    --max_compile_attempts 3 \
    --max_run_attempts 3 \
    $EXTRA_ARGS

echo ""
echo "=== [2/2] cuda -> omp (17 kernels) ==="
conda run -n unipar python datasets_after_eval_compile_main.py \
    --dataset_dir "$DATASET" \
    --from_api cuda \
    --to_api omp \
    --max_compile_attempts 3 \
    --max_run_attempts 3

echo ""
echo "======================================================"
echo " Done. Total: 34 kernels processed."
echo "======================================================"
