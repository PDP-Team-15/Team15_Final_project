#!/bin/bash
# 重置 dataset，讓 compile+run agent 可以從頭再跑
# 用法: ./reset_dataset.sh <dataset_dir> <direction>
# 例如: ./reset_dataset.sh /home/pdp15/UniPar_AI/Datasets/eval/deepseek_32b_baseline_shot_0_m_8192_t_0.2_p_0.9 cuda-omp

set -e

DATASET_DIR="$1"
DIRECTION="$2"   # e.g., cuda-omp or omp-cuda

if [ -z "$DATASET_DIR" ] || [ -z "$DIRECTION" ]; then
    echo "Usage: $0 <dataset_dir> <direction>"
    echo "  direction: cuda-omp or omp-cuda"
    exit 1
fi

SRC_DIR="$DATASET_DIR/HeCBench-$DIRECTION/src"

if [ ! -d "$SRC_DIR" ]; then
    echo "Error: $SRC_DIR does not exist"
    exit 1
fi

echo "Resetting kernels in $SRC_DIR ..."

for kernel_dir in "$SRC_DIR"/*/; do
    kernel=$(basename "$kernel_dir")
    [ "$kernel" = "src" ] && continue
    [ ! -d "$kernel_dir" ] && continue

    # 找出 pred_0 檔案（可能是 .cpp 或 .cu）
    pred0=""
    for f in "$kernel_dir"/pred_0.cpp "$kernel_dir"/pred_0.cu; do
        [ -f "$f" ] && pred0="$f" && break
    done

    if [ -z "$pred0" ]; then
        echo "  [SKIP] $kernel: no pred_0 found (already clean)"
        continue
    fi

    # 從 Makefile 找出 source 檔名
    makefile=""
    for mf in "$kernel_dir/Makefile.aomp" "$kernel_dir/Makefile"; do
        [ -f "$mf" ] && makefile="$mf" && break
    done

    source_file=""
    if [ -n "$makefile" ]; then
        source_file=$(grep -i "^source\s*=" "$makefile" | head -1 | sed 's/.*=\s*//' | tr -d '[:space:]')
    fi

    # 把 source 檔案還原成 pred_0 的內容
    if [ -n "$source_file" ] && [ -f "$kernel_dir/$source_file" ]; then
        cp "$pred0" "$kernel_dir/$source_file"
        echo "  [RESTORE] $kernel: $source_file <- pred_0"
    fi

    # 刪除所有 agent 生成的檔案
    rm -f "$kernel_dir"/pred_0.cpp "$kernel_dir"/pred_0.cu
    rm -f "$kernel_dir"/pred_compile_*.cpp "$kernel_dir"/pred_compile_*.cu
    rm -f "$kernel_dir"/pred_run_*.cpp "$kernel_dir"/pred_run_*.cu
    rm -f "$kernel_dir"/error_compile_*.txt "$kernel_dir"/error_run_*.txt
    rm -f "$kernel_dir"/result.txt
    rm -f "$kernel_dir"/output_run_*.txt
    rm -f "$kernel_dir"/run_error.txt
    rm -f "$kernel_dir"/compilation_error.txt
    rm -f "$kernel_dir"/*.bak
    rm -f "$kernel_dir"/*.o
    rm -f "$kernel_dir"/main "$kernel_dir"/a.out

    # 刪除 run attempt 目錄
    rm -rf "$kernel_dir"/run_[0-9]*/

    echo "  [DONE]    $kernel"
done

echo ""
echo "Reset complete. You can now re-run the agent."
