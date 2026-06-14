#!/bin/bash

source ~/.bashrc
source /usr/local/anaconda3/etc/profile.d/conda.sh

conda activate unipar

# echo "start with one"
if [[ $1 == "gpt" ]]; then
    SCRIPT="./llm_evaluation/new_gpt_inference.py"
else
    # Start the Llama vLLM server and wait for it to be ready.
    rm -f ./nohup.out
    ./llm_evaluation/deploy_2.sh &

    while [[ ! -f ./nohup.out ]]; do
        sleep 1
    done
    while ! grep -q "Starting vLLM API server on http://" ./nohup.out; do
        sleep 5
    done
    SCRIPT="./llm_evaluation/llama_inference_2.py"
fi

python -u $SCRIPT --num_shots 0 --temp 0.2 --top_p 0.9 --max_token 4096
echo "done with one"
# python -u $SCRIPT --num_shots 1 --temp 0.2 --top_p 0.9 --max_token 15000 