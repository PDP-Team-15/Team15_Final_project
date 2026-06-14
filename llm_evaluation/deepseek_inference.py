from openai import OpenAI
import os
import logging
import argparse
import json
import sys
from datetime import datetime
from tqdm import tqdm
import time

sys.path.append('data')#'../../data')
from kernel_dataset import KernelDataset

# Resolve paths relative to the repo root (one level above this file).
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_PATH = os.path.join(PROJECT_PATH, 'data', 'Datasets')
# DATASET_NAME = 'llama3.1_8b_eval' #change name for future reference when changing model
# DATASET_NAME = 'llama3.1_8b_test' 

def build_message(from_api, to_api, code, prompts, num_shots=0, prompt_style='baseline'):
    # prompt_kernels = ['accuracy', 'swish','permute']#, 'gabor'
    prompt_kernels = ['accuracy', 'gabor', 'permute']

    if prompt_style == 'strict':
        if from_api == 'cuda' and to_api == 'omp':
            messages = [
                {"role": "system", "content": "You are an HPC expert specializing in translating CUDA code to OpenMP. Your goal is to produce syntactically correct OpenMP code that compiles on the first attempt without any errors. Follow these guidelines:\n1. Remove all CUDA-specific constructs and qualifiers such as `__global__`, `__device__`, `__host__`, `__forceinline__`, and any CUDA-specific type qualifiers.\n2. Replace CUDA memory management functions (`cudaMalloc`, `cudaFree`, `cudaMemcpy`) with standard C functions (`malloc`, `free`).\n3. Replace CUDA kernel launches with direct function calls or OpenMP parallel regions using `#pragma omp parallel for`.\n4. Replace CUDA-specific variables (`blockIdx`, `blockDim`, `threadIdx`) with OpenMP thread management functions like omp_get_num_threads() and `omp_get_thread_num()`.\n5. Remove any CUDA-specific includes like #include <cuda.h> and ensure only necessary OpenMP includes like #include <omp.h> are present.\n6. Ensure all function parameters and return types are compatible with OpenMP and remove any unused parameters.\n7. Check and ensure all variables and function signatures match exactly between CUDA source and OpenMP implementation.\n8. Use standard C functions for any mathematical or utility operations that do not require OpenMP-specific handling.\n9. Output only the translated code without any comments, explanations, or debugging statements."},
                {"role": "user", "content": f"For each CUDA kernel code provided, systematically and accurately translate it to OpenMP. Include all necessary OpenMP headers and ensure correct function replacements. Remove any CUDA-specific code. Output a complete, functional OpenMP implementation ready for direct compilation without errors. Pay special attention to proper use of OpenMP functions, correct thread and task constructs, and appropriate memory management. Do not retain any CUDA-specific function calls or variables. Include all required headers at the beginning of the code. Ensure the resulting code is syntactically correct, properly formatted, and compiles without errors on the first attempt."},
            ]
        else:  # omp -> cuda
            messages = [
                {"role": "system", "content": "You are an HPC expert specializing in translating between parallel programming APIs. You will translate OMP code to CUDA code with the following requirements:\n\n1. Include all necessary CUDA headers (e.g., cuda_runtime.h, etc.)\n2. Replace OMP function calls with correct CUDA equivalents\n3. Use proper CUDA syntax for kernel launches and memory management\n4. Ensure all variables are correctly declared and accessible within CUDA context\n5. Function names must match the original code\n6. Include complete code only - no partial lines or commented code\n7. Output must compile without errors on first attempt\n8. CUDA kernel functions must be marked with __global__ \n9. Helper functions must be marked with __device__\n10. Include all necessary includes and preprocessor directives\n11. Ensure proper memory alignment and indexing\n12. Include complete error checking for CUDA API calls (if applicable)\n13. Use standard CUDA best practices for code structure and style\n14. All code must be syntactically complete - no missing semicolons or parenthesis\n15. Preserve the original algorithm's logic and functionality\n\nOutput only the translated code inside a code block. No explanations, no comments, no omissions. Ensure all CUDA-specific syntax is properly formatted and complete."},
                {"role": "user", "content": f"For each kernel code provided, translate it from {from_api} to {to_api}. Provide the complete code in {to_api}. Do not truncate or use ellipses. Ensure correctness. Do not add any additional comments or explanations."},
            ]
    # hints

    else: # baseline
        messages = [
            {"role": "system", "content": "You are an HPC expert specializing in translating between parallel programming APIs."},
            {"role": "user", "content": f"For each kernel code provided, translate it from {from_api} to {to_api}. Provide the complete code in {to_api}. Do not truncate or use ellipses. Do not change the main function Ensure correctness. All function names must match."},
        ]
    
    for kernel in prompt_kernels[:num_shots]:
        from_code = prompts.dataset[f"{kernel}-{from_api}"]
        to_code = prompts.dataset[f"{kernel}-{to_api}"]

        messages.append({"role": "user", "content": f'Translate the following code  from {from_api} to {to_api}:\n{from_code}'})
        messages.append({"role": "assistant", "content": f'Here is the translated code:\n{to_code}'})

    messages.append({"role": "user", "content": f'Translate the following code  from {from_api} to {to_api}:\n{code}'})
    return messages


def translate_kernels(args, client, kernels, prompts, batch_size=4, num_return_sequences=3, output_dir=None):
    timer = 0
    direction = getattr(args, 'direction', 'both')
    batches = [kernels[i:i+batch_size] for i in range(0, len(kernels), batch_size)]
    filtered = [b for b in batches if
                (direction in ('both', 'omp_cuda') and b[0][1] == 'omp' and b[0][3] == 'cuda') or
                (direction in ('both', 'cuda_omp') and b[0][1] == 'cuda' and b[0][3] == 'omp')]
    # for i in range(0, len(kernels), batch_size):
    for batch in tqdm(filtered, desc="Translating", unit="kernel"):
        # original
        # batch = kernels[i:i+batch_size]
        # if batch[0][1] not in ['serial', 'omp', 'cuda'] or batch[0][3] not in ['serial', 'omp', 'cuda']:
        #     continue
        
        batch_messages = [build_message(kernel[1], kernel[3], kernel[2], prompts, num_shots=args.num_shots, prompt_style=args.prompt_style) for kernel in batch]
        kernel = batch[0]
        kernel_name, from_api, from_code, to_api, to_code = kernel
        try:
            outputs_batch = client.chat.completions.create(
                model='Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ',#change name here to change the model in huggingface
                messages=batch_messages[0],
                max_tokens=args.max_token,
                temperature=args.temp,
                top_p=args.top_p,
                n=num_return_sequences,
                timeout=1500,
            )
        except Exception as e:
            timer +=1
            logging.error(f"Error {kernel_name}: {from_api}->{to_api}, {timer}, error :  {e}")
            if timer % 15 == 0:
                time.sleep(30)
            if timer > 50:
                logging.error("Too many errors. Exiting now.")
                exit(1)
            continue
        else:
            timer = 0

        # output_dir = str(output_dir)
        if output_dir is None:
            logging.error("Output directory not provided.")
            return
        output_path = os.path.join(output_dir, f'{kernel_name}_{from_api}_{to_api}')
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, 'msg.cpp'), 'w') as f:
            f.write("\n".join([msg["content"] for msg in batch_messages[0]]))

        with open(os.path.join(output_path, 'truth.cpp'), 'w') as f_truth:
            f_truth.write(to_code)

        for idx, output in enumerate(outputs_batch.choices):
            with open(os.path.join(output_path, f'pred_{idx}.cpp'), 'w') as f_pred:
                f_pred.write(output.message.content)

            logging.info(f"Processed {kernel_name}: {from_api}->{to_api}")

def get_directory_names(path):
    # Check if the path exists
    if not os.path.exists(path):
        return []
    return [item for item in os.listdir(path) 
            if os.path.isdir(os.path.join(path, item))]


if __name__ == '__main__':
    dataset_dir = os.path.join(PROJECT_PATH, 'data/Datasets/HeCBench')#points to where the dataset jsonl 
    to_ignore = []

    parser = argparse.ArgumentParser(description="Few-Shot Learning Argument Parser")
    parser.add_argument('--num_shots', type=int, required=True, help='Number of shots for few-shot learning')
    parser.add_argument('--temp', type=float, required=True)
    parser.add_argument('--top_p', type=float, required=True)
    parser.add_argument('--max_token', type=int, required=True)
    parser.add_argument('--overwrite_out', type=bool, required=False)
    parser.add_argument('--port', type=int, required=False, default=8001)
    parser.add_argument('--dataset_name', type=str, default='llama3.1_8b_eval')
    parser.add_argument('--prompt_style', type=str, default='baseline',choices=['baseline', 'strict', 'hints', 'cot'], help='Prompt style to use for translation')
    parser.add_argument('--direction', type=str, default='both', choices=['both', 'omp_cuda', 'cuda_omp'], help='Translation direction to run')

    args = parser.parse_args()
    DATASET_NAME = f"{args.dataset_name}_{args.prompt_style}"

    # Set up logging
    current_time = datetime.now().strftime("%d-%m-%y_%H-%M")
    log_file_name = f'few_shot={args.num_shots}_{DATASET_NAME}_{current_time}_pid={os.getpid()}.log'
    print("logfile_name:",log_file_name)
    logging.basicConfig(filename=log_file_name, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load dataset
    dataset_dir = os.path.join(PROJECT_PATH, 'data/Datasets/HeCBench')
    out_path = os.path.join(DATASET_PATH, f'vllm_{DATASET_NAME}_shots={args.num_shots}_max_token={args.max_token}_temp={args.temp}_p={args.top_p}_port={args.port}')
    if not args.overwrite_out:
        to_ignore = get_directory_names(out_path)
        print(type(to_ignore))
    else:
        print("overwriting output directory")
        to_ignore = []
    # test_set = KernelDataset(dataset_dir, dataset_type='dataset_Data_encoding_decoding',ignore_kernels=to_ignore)#filter out already done in this run here
    test_set = KernelDataset(dataset_dir, dataset_type='test', ignore_kernels=to_ignore) # all test set data
    logging.info(f'Dataset size: {len(test_set)}')

    # Set up few-shot prompts
    prompt_set = KernelDataset(dataset_dir, dataset_type='prompt_Data_encoding_decoding')

    client = OpenAI(
        base_url=f'http://192.168.50.212:{args.port}/v1',
        api_key='token123'
    )

    # Process the kernels in batches
    translate_kernels(args, client, test_set, prompt_set, batch_size=1, num_return_sequences=1, output_dir=out_path)
    logging.info("Finished processing all kernels.")
    print("Finished processing all kernels.")
