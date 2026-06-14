import os
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
lengths = []

file_path = "/home/pdp15/UniPar_AI/data/Datasets/HeCBench/train.jsonl"
with open(file_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        # Assuming the structure is something we can extract the full text from
        if 'kernel_name' in data and 'code' in data:
            # Reconstruct the approximate prompt as done in tokenize_function
            from_api = data.get('kernel_api', '')
            to_api = "some_target" # Dummy target for length check
            
            # The code is usually a dictionary of filenames to source code
            from_code = ""
            for filename, code in data['code'].items():
                from_code += code
                
            text = f"You are an HPC expert specializing in translating between parallel programming APIs. Translate the following kernel from {from_api} to {to_api}. Provide the complete code in {to_api}. Do not truncate or use ellipses. Ensure correctness. \nCode:\n {from_code}"
            # Also need to account for output length, let's just double it for a very conservative estimate
        elif 'instruction' in data and 'response' in data:
            text = data['instruction'] + "\n" + data['response']
        else:
            # fallback
            text = str(data)
            
        tokens = tokenizer.encode(text)
        lengths.append(len(tokens))

if lengths:
    lengths.sort()
    print(f"Total entries: {len(lengths)}")
    print(f"Min length: {lengths[0]}")
    print(f"Median length: {lengths[len(lengths)//2]}")
    print(f"90th percentile: {lengths[int(len(lengths)*0.9)]}")
    print(f"95th percentile: {lengths[int(len(lengths)*0.95)]}")
    print(f"99th percentile: {lengths[int(len(lengths)*0.99)]}")
    print(f"Max length: {lengths[-1]}")
    print(f"\nHow many > 4096: {sum(1 for l in lengths if l > 4096)}")
    print(f"How many > 8192: {sum(1 for l in lengths if l > 8192)}")
