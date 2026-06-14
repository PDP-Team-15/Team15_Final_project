#!/bin/bash
if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    # Default to the second GPU if none is specified.
    export CUDA_VISIBLE_DEVICES=1
fi

LOG_FILE=./nohup_qwen.out
echo '################################ deploy_qwen_2 ##########################################' > "$LOG_FILE"

echo "running vllm server (qwen)" > ./vllm_server_status_qwen.txt

nohup vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
    --enable-chunked-prefill False \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --swap-space 8 \
    --api-key token123 \
    --port 8003 \
    > "$LOG_FILE" 2>&1 &

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "crashed:$EXIT_CODE" >> ./vllm_server_status_qwen.txt
else
    echo "stopped:$EXIT_CODE" >> ./vllm_server_status_qwen.txt
fi
