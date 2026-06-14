#!/bin/bash
if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    export CUDA_VISIBLE_DEVICES=0
fi

LOG_FILE=./nohup_14b.out
echo '################################ deploy_14b ##########################################' > "$LOG_FILE"

echo "running vllm server (qwen14b)" > ./vllm_server_status_14b.txt

nohup vllm serve Qwen/Qwen2.5-Coder-14B-Instruct-AWQ \
    --quantization awq \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --swap-space 8 \
    --api-key token123 \
    --port 8004 \
    > "$LOG_FILE" 2>&1 &

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "crashed:$EXIT_CODE" >> ./vllm_server_status_14b.txt
else
    echo "stopped:$EXIT_CODE" >> ./vllm_server_status_14b.txt
fi
