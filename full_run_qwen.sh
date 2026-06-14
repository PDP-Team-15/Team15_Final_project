#!/bin/bash

source ~/.bashrc
source /usr/local/anaconda3/etc/profile.d/conda.sh

conda activate unipar

# Start the Qwen vLLM server and wait for it to be ready.
rm -f ./nohup_qwen.out
./llm_evaluation/deploy_qwen_2.sh &

while [[ ! -f ./nohup_qwen.out ]]; do
    sleep 1
done
while ! grep -q "Starting vLLM API server on http://" ./nohup_qwen.out; do
    sleep 5
done

python -u ./llm_evaluation/qwen_inference_2.py --num_shots 0 --temp 0.2 --top_p 0.9 --max_token 4096

echo "done with one"
