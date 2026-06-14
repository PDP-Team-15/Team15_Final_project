#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.6"

# 下面接你原本的 nohup vllm serve...
if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    export CUDA_VISIBLE_DEVICES=1
fi

echo '################################ deploy_finetune ##########################################' > nohup.out

echo "running vllm server" > ./vllm_server_status.txt

# nohup vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct\ #\-FP8
#neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8
    # --uvicorn-log-level debug \
    # --disable_custom_all_reduce \
# nohup vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct \
# port 8002 for GPU:0
# port 8001 for GPU:1

export CUDA_VISIBLE_DEVICES=1 # for GPU:1
model="llm_finetune/ft_llama8"

nohup vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-modules Llama-3.1-8B-Instruct_Lora=${model} \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --api-key token123 \
    --port 8005 &

# --quantization="fp8" \
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "crashed:$EXIT_CODE" >> ../vllm_server_status.txt
else
    echo "stopped:$EXIT_CODE" >> ../vllm_server_status.txt
fi


### prev version - transformers 4.48.2
### prev version - vllm 0.7.2
