# This repository is a derivative work originating from UniPar_AI.
git clone https://github.com/Scientific-Computing-Lab/UniPar_AI.git

## Requirements

- conda environment: `unipar`
- HeCBench source code: `~/HeCBench/` (must be cloned)
- GPU: at least 1 × RTX 3090 (24GB VRAM)

---

## One-Time Setup

```bash
conda activate unipar
cd ~/UniPar_AI
ln -s ~/HeCBench/src ~/UniPar_AI/data/Datasets/HeCBench/src
```

---

## Step 1: Deploy vLLM Server

Choose the model to deploy:

### LLaMA 3.1 8B (GPU 0, port 8002)

```bash
bash llm_evaluation/deploy_2.sh
```

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| GPU | `CUDA_VISIBLE_DEVICES=0` |
| Port | `8002` |
| Max context | `8192 tokens` |

### Qwen2.5-Coder-7B (GPU 1, port 8003)

```bash
bash llm_evaluation/deploy_qwen_2.sh
```

| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| GPU | `CUDA_VISIBLE_DEVICES=1` |
| Port | `8003` |
| Max context | `8192 tokens` |

### Qwen2.5-Coder-14B AWQ (GPU 1, port 8004)

```bash
bash scripts/deploy_qwen_14b.sh
```

| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ` |
| GPU | `CUDA_VISIBLE_DEVICES=0` |
| Port | `8004` |
| Max context | `32768 tokens` |
| Quantization | AWQ 4-bit (~8GB VRAM) |

Monitor server startup:

```bash
tail -f nohup.out          # LLaMA
tail -f nohup_qwen.out     # Qwen 7B
tail -f nohup_14b.out      # Qwen 14B
```

Wait until `Application startup complete.` appears before running inference.

---

## Step 2: Run Inference

### LLaMA 3.1 8B

```bash
bash scripts/run_llama.sh <temp> <top_p> <max_token> <port> [prompt_style]
```

```bash
# Example (strict prompt, port 8002)
bash scripts/run_llama.sh 0.2 0.9 2048 8002 strict
```

### Qwen2.5-Coder-7B

```bash
bash scripts/run_qwen.sh <temp> <top_p> <max_token> <port> [prompt_style]
```

```bash
# Example (strict prompt, port 8003)
bash scripts/run_qwen.sh 0.2 0.9 2048 8003 strict
```

### Qwen2.5-Coder-14B AWQ

```bash
bash scripts/run_qwen.sh <temp> <top_p> <max_token> <port> [prompt_style] [model_name]
```

```bash
# Example (strict prompt, port 8004)
bash scripts/run_qwen.sh 0.2 0.9 4096 8004 strict Qwen/Qwen2.5-Coder-14B-Instruct-AWQ
```

**Parameters:**

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `temp` | Temperature | `0.2` |
| `top_p` | Top-p sampling | `0.9` |
| `max_token` | Max output tokens | `2048` (7B) / `4096` (14B) |
| `port` | vLLM server port | `8002` / `8003` / `8004` |
| `prompt_style` | `baseline` or `strict` | `strict` |
| `model_name` | (Qwen only) model identifier | `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ` |

**Prompt styles:**
- `baseline` — standard translation prompt
- `strict` — forces `\`\`\`cpp ... \`\`\`` output format, reduces incomplete code

Output is saved to:
```
data/Datasets/vllm_<dataset_name>_<prompt_style>_shots=0_max_token=<max_token>_temp=<temp>_p=<top_p>/
```

Inference only runs `omp → cuda` and `cuda → omp` translations.

Monitor progress:
```bash
tail -f few_shot=0_<model>_<timestamp>_pid=<pid>.log
```

---

## Step 3: Compile Evaluation

Edit `scripts/run_compile.sh` to set `MODEL_OUT` and `EVAL_DIR` to match the inference output:

```bash
# scripts/run_compile.sh
MODEL_OUT="vllm_qwen2.5_coder_14b_test_strict_shots=0_max_token=4096.0_temp=0.2_p=0.9"
EVAL_DIR="qwen2.5_coder_14b_test_strict_shot_0_m_4096_t_0.2_p_0.9"
```

Then run:

```bash
bash scripts/run_compile.sh
```

Compiled results are saved to:
```
Datasets/eval/<EVAL_DIR>/HeCBench-<from_api>-<to_api>/
```

---

## Step 4: Run & Speedup Evaluation

Edit `scripts/eval.sh` to set `--run_names` and `--log` to match the eval directory:

```bash
# scripts/eval.sh
python eval/run_and_speedup/eval_run_new.py \
    --target cuda omp \
    --from_apis omp cuda \
    --run_names "deepseek_32b_strict_v2_11k_strict_shot_0_m_11000_t_0.2_p_0.9" \
    --log Datasets/eval/deepseek_32b_strict_v2_11k_strict_shot_0_m_11000_t_0.2_p_0.9/eval_run.log
```

Then run:

```bash
bash scripts/eval.sh
```

Results are saved to:
```
Datasets/eval/<run_name>/eval_run.log     # compile errors + run output
Datasets/eval/<run_name>/eval_results.json  # compile rate / correct rate per pair
```

**Example `eval_results.json`:**
```json
{
  "per_pair": {
    "omp-cuda": { "compile": "5/18", "compile_rate": 0.278, "correct": "3/18", "correct_rate": 0.167 },
    "cuda-omp": { "compile": "5/17", "compile_rate": 0.294, "correct": "0/17", "correct_rate": 0.0 }
  }
}
```

---

## Step 5 (Optional): GEPA Prompt Optimization

GEPA (Genetic-Pareto) automatically evolves the `system_prompt` and `user_instruction`
used for zero-shot translation, replacing the fixed hand-written prompts with candidates
optimized for first-attempt compilation success. A *reflection LLM* reads compiler error
traces and proposes revised prompts; GEPA maintains a Pareto frontier of candidates
evaluated over a validation subset drawn from the training split. The held-out test set
is never touched.

**Script:** `multiagent_pipeline/gepa_optimize_prompts.py`

### Train / Val Split

| Pool | Flag | Role |
|------|------|------|
| Feedback pool | `--num_train` | Kernels sampled for minibatch reflection — the reflection LLM reads their error traces to propose new prompt candidates |
| Pareto validation set | `--num_val` | Kernels used to score and rank candidates on the Pareto frontier |

Both pools are drawn from `train.jsonl`. `test.jsonl` is never used during optimization.

### Prerequisites

- The **translator model** (the model being optimized) must be running as an OpenAI-compatible server — e.g., a local vLLM instance on `--translator_url`.
- The **reflection model** can be a hosted cloud API (LiteLLM resolves it automatically, no extra server needed) or a second local vLLM endpoint set via `--reflection_base_url`.

### Usage

```bash
cd /home/pdp && conda run -n unipar python UniPar_AI/multiagent_pipeline/gepa_optimize_prompts.py \
    --hecbench_omp_dir "UniPar_AI/data/Datasets/HeCBench" \
    --from_api omp --to_api cuda \
    --num_train 12 --num_val 6 \
    --max_metric_calls 60 \
    --max_tokens 4096 \
    --translator_url http://localhost:8006/v1 \
    --translator_model Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ \
    --translator_api_key token123 \
    --reflection_lm "openai/Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ" \
    --reflection_base_url http://localhost:8006/v1 \
    --reflection_api_key token123 \
    --no_parallel
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--hecbench_omp_dir` | `data/Datasets/HeCBench` | Path to HeCBench directory containing `src/kernel-api/` subdirectories with Makefiles. Required for compilation scoring. |
| `--dataset_dir` | same as `--hecbench_omp_dir` | Path to directory containing `train.jsonl` and `test.jsonl`. |
| `--from_api` | `omp` | Source parallel API. Choices: `serial`, `omp`, `cuda`, `hip`, `sycl`. |
| `--to_api` | `cuda` | Target parallel API. Choices: `omp`, `cuda`. |
| `--num_train` | `30` | Size of the feedback pool (minibatch reflection sampling). |
| `--num_val` | `20` | Size of the Pareto validation set (candidate scoring). |
| `--max_metric_calls` | `100` | Total evaluation budget — each call translates and compiles one kernel. Controls how long optimization runs. |
| `--translator_url` | `http://localhost:8004/v1` | Base URL of the vLLM server for the **model being optimized**. |
| `--translator_model` | `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ` | Model name as registered by the vLLM server. |
| `--translator_api_key` | `token123` | API key for the translation server. |
| `--reflection_lm` | `openai/gpt-4o` | LiteLLM model string for the **reflection model** — the LLM that reads error traces and proposes improved prompts. The `openai/` prefix is a LiteLLM routing prefix (not a vendor restriction); use it with `--reflection_base_url` to point at a local vLLM server (e.g. `"openai/DeepSeek-R1-..."` + `--reflection_base_url http://localhost:8006/v1`). Without `--reflection_base_url`, LiteLLM routes to the cloud provider implied by the prefix. |
| `--reflection_base_url` | `None` | Base URL for a local reflection LM server. When set, overrides LiteLLM cloud routing. |
| `--reflection_api_key` | `$OPENAI_API_KEY` | API key for the reflection server (falls back to `OPENAI_API_KEY` env var for cloud APIs). |
| `--max_tokens` | `4096` | Max tokens for translation and error-fix outputs. Keep below the model's context limit minus prompt overhead. |
| `--output_dir` | `output/gepa_<timestamp>` | Directory for run logs and the optimized candidate JSON. Created automatically. |
| `--no_parallel` | `False` | Disable parallel kernel evaluation. Use when vLLM memory is constrained. |

### Output

The script writes `optimized_candidate.json` to `--output_dir`:

```json
{
  "best_val_score": 0.85,
  "best_candidate": {
    "system_prompt": "...",
    "user_instruction": "..."
  },
  "seed_candidate": { "system_prompt": "...", "user_instruction": "..." },
  "config": { "from_api": "omp", "to_api": "cuda", "num_train": 12, "num_val": 6, ... }
}
```

To use the optimized prompts, copy `best_candidate.system_prompt` and
`best_candidate.user_instruction` into `multiagent_pipeline/questioner.py`,
replacing the corresponding hand-written prompt strings.
