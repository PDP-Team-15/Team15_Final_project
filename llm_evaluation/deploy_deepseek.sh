#!/bin/bash
# if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
#     AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '{print $1}' | paste -sd "," -)
#     export CUDA_VISIBLE_DEVICES=$AVAILABLE_GPUS
# fi

export HF_HOME=/mnt/sda_path/huggingface_cache

LOG_FILE=./nohup_deepseek.out
echo '################################ deploy_deepseek ##########################################' > "$LOG_FILE"

echo "running vllm server (deepseek)" > ./vllm_server_status_deepseek.txt

export CUDA_VISIBLE_DEVICES=0,1

nohup vllm serve Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ \
    --tensor-parallel-size 2 \
    --quantization awq \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.80 \
    --max-num-seqs 64 \
    --swap-space 4 \
    --api-key token123 \
    --port 8006 \
    > "$LOG_FILE" 2>&1 &

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "crashed:$EXIT_CODE" >> ./vllm_server_status_deepseek.txt
else
    echo "stopped:$EXIT_CODE" >> ./vllm_server_status_deepseek.txt
fi
