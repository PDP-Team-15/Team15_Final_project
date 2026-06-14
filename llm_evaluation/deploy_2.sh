#!/bin/bash
if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '{print $1}' | paste -sd "," -)
    export CUDA_VISIBLE_DEVICES=$AVAILABLE_GPUS
fi
echo '################################ deploy_2 ##########################################' > nohup.out

echo "running vllm server" > ./vllm_server_status.txt

# nohup vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct\ #\-FP8
#neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8
    # --uvicorn-log-level debug \
    # --disable_custom_all_reduce \
# nohup vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct \
# port 8002 for GPU:0
# port 8001 for GPU:1


export CUDA_VISIBLE_DEVICES=1
nohup vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --host 0.0.0.0 \
    --port 8009 \
    --gpu-memory-utilization 0.85 \
    --swap-space 8 \
    --api-key token123 &

# --quantization="fp8" \
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "crashed:$EXIT_CODE" >> ../vllm_server_status.txt
else
    echo "stopped:$EXIT_CODE" >> ../vllm_server_status.txt
fi


### prev version - transformers 4.48.2
### prev version - vllm 0.7.2
