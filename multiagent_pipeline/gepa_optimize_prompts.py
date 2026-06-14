"""
Prompt optimization for UniPar_AI using GEPA's optimize_anything API.

Optimizes the system_prompt and user_instruction used to translate HPC kernels
between parallel APIs. GEPA reads compiler error messages as Actionable Side
Information (ASI) and evolves prompts that produce cleaner, first-attempt
compilable translations.

Scoring:
  1.0  — compiled without any model-assisted fix
  0.5  — compiled after 1 model-assisted fix attempt
  0.0  — failed to compile

Usage:
    python gepa_optimize_prompts.py \
        --dataset_dir data/Datasets/HeCBench \
        --hecbench_omp_dir data/Datasets/HeCBench \
        --from_api omp --to_api cuda \
        --num_train 30 --num_val 20 \
        --max_metric_calls 100 \
        --reflection_lm openai/gpt-4o
        
    In pdp Dir: 
    
    python UniPar_AI/multiagent_pipeline/gepa_optimize_prompts.py \
        --hecbench_omp_dir "UniPar_AI/Datasets/eval/deepseek_32b_strict_shot_0_m_8192_t_0.2_p_0.9/HeCBench-omp-cuda" \
        --from_api omp \
        --to_api cuda \
        --num_train 12 \
        --num_val 6 \
        --max_metric_calls 60 \
        --translator_url http://localhost:8006/v1 \
        --translator_model "Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ" \
        --translator_api_key token123 \
        --reflection_lm "openai/Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ" \
        --reflection_base_url http://localhost:8006/v1 \
        --reflection_api_key token123 \
        --no_parallel
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Local imports from multiagent_pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gepa.optimize_anything as oa
from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig, optimize_anything

from cuda_dataset import KernelDataset
from model_agent import ModelAgent
from utils import extract_code_from_model_output

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Seed prompts — current handwritten prompts from questioner.py
SEED_CANDIDATE = {
    "system_prompt": (
        "You are an HPC expert specializing in translating between parallel programming APIs. "
        "Translate precisely and completely. Output only the translated code inside a code block. "
        "No explanations, no comments, no omissions."
    ),
    "user_instruction": (
        "For each kernel code provided, translate it from {from_api} to {to_api}. "
        "Provide the complete code in {to_api}. Do not truncate or use ellipses. "
        "Ensure correctness. Do not add any additional comments or explanations."
    ),
}

# Helpers
def _extract_code_string(code) -> str:
    """KernelDataset stores code as a single-key dict {filename: content}."""
    if isinstance(code, dict):
        return next(iter(code.values()))
    return str(code)


def _build_messages(candidate: dict, from_api: str, to_api: str, source_code: str) -> list[dict]:
    """Build chat messages from the candidate prompts."""
    system_prompt = candidate["system_prompt"]
    # Use plain replace instead of format_map: GEPA may inject code snippets
    # containing {blockIdx.x} or similar CUDA expressions that break format_map.
    user_instruction = (
        candidate["user_instruction"]
        .replace("{from_api}", from_api)
        .replace("{to_api}", to_api)
    )

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"{user_instruction}\n\n"
                f"Translate the following code from {from_api} to {to_api}. "
                f"Provide the complete code in {to_api}:\n{source_code}"
            ),
        },
        {
            "role": "assistant",
            "content": f"```{to_api}\n// Translated code goes here\n```",
        },
    ]


def _copy_build_files(
    hecbench_kernel_dir: str, build_dir: str, to_api: str, target_filename: str = ""
) -> bool:
    """
    Copy Makefiles and symlink supporting files from the HeCBench source into build_dir.
    The caller will write the translated code separately as `target_filename`.

    target_filename: the actual source filename the Makefile compiles (e.g. 'ao.cu',
        'stencil_1d.cu', or 'main.cu').  That file is skipped here so the caller's
        translated code is not overwritten by a symlink of the original.
    Returns True if the hecbench_kernel_dir exists and files were copied.
    """
    if not os.path.isdir(hecbench_kernel_dir):
        return False

    # Resolve to absolute so symlinks work correctly from any temp dir under /tmp
    hecbench_kernel_dir = os.path.abspath(hecbench_kernel_dir)

    # Files to skip (caller writes the translated version)
    skip_files = {"main.cpp", "main.cu"}
    if target_filename:
        skip_files.add(target_filename)

    for filename in os.listdir(hecbench_kernel_dir):
        src = os.path.join(hecbench_kernel_dir, filename)
        dst = os.path.join(build_dir, filename)

        if os.path.exists(dst) or os.path.islink(dst):
            os.unlink(dst)

        name_lower = filename.lower()
        is_makefile = name_lower == "makefile" or filename.startswith("Makefile")

        if is_makefile:
            shutil.copy2(src, dst)
            # Fix relative paths in Makefile so they resolve from build_dir
            with open(dst) as f:
                content = f.read()
            import re
            content = re.sub(
                r"\.\./([^\s:=\"']+)",
                lambda m: str(os.path.join(hecbench_kernel_dir, m.group(1))),
                content,
            )
            with open(dst, "w") as f:
                f.write(content)
        elif filename not in skip_files:
            os.symlink(src, dst)

    return True


def _run_make(build_dir: str, to_api: str) -> tuple[bool, str]:
    """Run make in build_dir for the given target API. Returns (success, error_output)."""
    try:
        subprocess.run(
            ["make", "clean"],
            cwd=build_dir,
            capture_output=True,
            text=True,
        )
    except Exception:
        pass

    if to_api == "omp":
        cmd = ["make", "-f", "Makefile.aomp", "DEVICE=cpu"]
    else:
        cmd = ["make"]

    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    return result.returncode == 0, result.stderr


def _fix_and_recompile(
    code: str,
    error_msg: str,
    build_dir: str,
    main_path: str,
    to_api: str,
    model_agent: ModelAgent,
) -> tuple[bool, str, str]:
    """
    Ask the model to fix compilation errors, write the fix, and recompile.
    Returns (success, final_code, final_error).
    """
    speciality = "CUDA" if to_api == "cuda" else "C++ (OpenMP)"
    messages = [
        {
            "role": "system",
            "content": (
                f"You are an expert {speciality} programmer specializing in HPC code. "
                "Fix compilation errors. Provide only the complete corrected code inside "
                "a proper code block. No explanations, no omissions, no placeholders."
            ),
        },
        {
            "role": "user",
            "content": (
                f"The following code fails to compile.\n\nCode:\n```\n{code}\n```\n\n"
                f"Compilation Error:\n```\n{error_msg}\n```\n\n"
                f"Please output only the complete corrected code inside:\n"
                f"```{to_api}\n<full corrected code here>\n```"
            ),
        },
        {
            "role": "assistant",
            "content": f"```{to_api}\n<full corrected code here>\n```",
        },
    ]

    try:
        response = model_agent.fixer_client.chat.completions.create(
            model=model_agent.fixer_model,
            messages=messages,
            max_tokens=model_agent.max_tokens,
            temperature=model_agent.temperature,
            top_p=model_agent.top_p,
        )
        fixed_code = extract_code_from_model_output(response.choices[0].message.content)
    except Exception as e:
        return False, code, f"Model fix failed: {e}"

    with open(main_path, "w") as f:
        f.write(fixed_code)

    success, error = _run_make(build_dir, to_api)
    return success, fixed_code, error


# GEPA evaluator
def make_evaluator(model_agent: ModelAgent, hecbench_omp_dir: str, max_metric_calls: int = 0):
    """
    Returns an evaluator function closed over model_agent and hecbench paths.

    Each `example` dict contains:
        kernel_name, from_api, to_api, source_code
    """
    import sys
    from tqdm import tqdm
    pbar = tqdm(total=max_metric_calls or None, desc="GEPA evals", unit="eval",
                bar_format="{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
                file=sys.stdout, dynamic_ncols=True)

    def evaluate(candidate: dict, example: dict) -> tuple[float, dict]:
        kernel_name = example["kernel_name"]
        from_api = example["from_api"]
        to_api = example["to_api"]
        source_code = example["source_code"]

        # -Translate using candidate prompts
        messages = _build_messages(candidate, from_api, to_api, source_code)
        try:
            response = model_agent.translator_client.chat.completions.create(
                model=model_agent.translator_model,
                messages=messages,
                max_tokens=model_agent.max_tokens,
                temperature=model_agent.temperature,
                top_p=model_agent.top_p,
            )
            translated_raw = response.choices[0].message.content
        except Exception as e:
            oa.log(f"[{kernel_name}] Translation API error: {e}")
            return 0.0, {"Error": str(e), "kernel": kernel_name}

        translated_code = extract_code_from_model_output(translated_raw)
        # Use the actual source filename from the dataset; some kernels compile
        # ao.cu, stencil_1d.cu, etc. — not necessarily main.cu/main.cpp.
        target_filename = example.get(
            "target_filename",
            "main.cu" if to_api == "cuda" else "main.cpp",
        )

        # Compile in a temp directory
        hecbench_kernel_dir = os.path.join(
            hecbench_omp_dir, "src", f"{kernel_name}-{to_api}"
        )

        with tempfile.TemporaryDirectory(prefix=f"gepa_{kernel_name}_") as build_dir:
            files_ready = _copy_build_files(
                hecbench_kernel_dir, build_dir, to_api, target_filename=target_filename
            )
            if not files_ready:
                oa.log(f"[{kernel_name}] No HeCBench source at {hecbench_kernel_dir}, skipping compile")
                pbar.set_postfix(kernel=kernel_name, score="skip")
                pbar.update(1)
                return 0.0, {"Error": f"Missing hecbench dir: {hecbench_kernel_dir}", "kernel": kernel_name}

            main_path = os.path.join(build_dir, target_filename)
            with open(main_path, "w") as f:
                f.write(translated_code)

            success, compile_error = _run_make(build_dir, to_api)

            if success:
                score = 1.0
                oa.log(f"[{kernel_name}] Compiled on first attempt ✓")
                pbar.set_postfix(kernel=kernel_name, score="1.0 ✓")
                pbar.update(1)
                return score, {
                    "kernel": kernel_name,
                    "from_api": from_api,
                    "to_api": to_api,
                    "compiled": True,
                    "fix_needed": False,
                    "source_snippet": source_code[:400],
                    "translated_snippet": translated_code[:400],
                }

            # One model-assisted fix attempt 
            oa.log(f"[{kernel_name}] Compile failed, trying model fix...")
            oa.log(f"[{kernel_name}] Compile error:\n{compile_error[:600]}")

            fixed_success, fixed_code, fixed_error = _fix_and_recompile(
                translated_code, compile_error, build_dir, main_path, to_api, model_agent
            )

            if fixed_success:
                score = 0.5
                oa.log(f"[{kernel_name}] Compiled after model fix ✓")
            else:
                score = 0.0
                oa.log(f"[{kernel_name}] Still failed after model fix ✗")
                oa.log(f"[{kernel_name}] Second error:\n{fixed_error[:400]}")

            pbar.set_postfix(kernel=kernel_name, score=f"{score:.1f}{'~' if fixed_success else '✗'}")
            pbar.update(1)
            return score, {
                "kernel": kernel_name,
                "from_api": from_api,
                "to_api": to_api,
                "compiled": fixed_success,
                "fix_needed": True,
                "initial_compile_error": compile_error[:600],
                "post_fix_error": fixed_error[:400] if not fixed_success else "",
                "source_snippet": source_code[:400],
                "translated_snippet": translated_code[:400],
            }

    return evaluate


# Dataset loading
def load_examples(
    dataset_dir: str,
    from_api: str,
    to_api: str,
    num_train: int,
    num_val: int,
    hecbench_omp_dir: str = "",
    shuffle_seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Load kernel examples from train.jsonl (NOT test.jsonl) so test.jsonl stays
    as a clean held-out set for final evaluation.

    Only kernels that satisfy all three conditions are included:
      1. Single source file for both from_api and to_api (avoids multi-file
         pipeline issues in eval_compile).
      2. Both {kernel}-{from_api} and {kernel}-{to_api} build directories exist
         under {hecbench_omp_dir}/src/ (otherwise GEPA scoring always returns 0).
      3. Not in COMPILE_FAIL_KERNELS — kernels whose Makefiles have cross-kernel
         path dependencies or missing headers that prevent compilation.

    Returns (trainset, valset) as non-overlapping lists of example dicts.
    """
    # Kernels confirmed to fail compilation due to structural Makefile issues
    # (cross-kernel ../ path deps, missing refdata headers, etc.) — verified
    # by a full compile-test pass over all 175 single-file train candidates.
    COMPILE_FAIL_KERNELS = {
        "bfs", "boxfilter", "conversion", "debayer", "diamond",
        "extend2", "gc", "interval", "kernelLaunch", "miniWeather",
        "openmp", "sad", "sptrsv", "testSNAP", "wordcount",
    }

    dataset = KernelDataset(
        dataset_dir,
        dataset_type="train",
        parallel_apis=[from_api, to_api],
        from_api=from_api,
        to_api=to_api,
    )

    src_base = os.path.join(hecbench_omp_dir, "src") if hecbench_omp_dir else ""

    all_examples = []
    for i in range(len(dataset)):
        name, api1, code1, api2, code2 = dataset[i]

        if name in COMPILE_FAIL_KERNELS:
            continue

        # Require single source file for both sides
        if isinstance(code1, dict) and len(code1) != 1:
            continue
        if isinstance(code2, dict) and len(code2) != 1:
            continue

        # Require build directories when hecbench_omp_dir is given
        if src_base:
            if not os.path.isdir(os.path.join(src_base, f"{name}-{api2}")):
                continue
            if not os.path.isdir(os.path.join(src_base, f"{name}-{api1}")):
                continue

        source_code = _extract_code_string(code1)

        # Skip kernels whose source code alone would overflow the model context.
        # Model max_model_len=16384; reserve 4096 for output + ~512 for prompt
        # overhead → source must fit in ≈11872 tokens ≈ 47488 chars (chars/4 est).
        if len(source_code) > 47000:
            logging.warning(f"Skipping {name}: source too long ({len(source_code)} chars)")
            continue

        # Derive the actual target source filename from the dataset.
        # Some kernels use non-standard names (e.g. ao.cu, stencil_1d.cu).
        # GEPA must write translated code to that filename so the Makefile
        # compiles the translation, not a stale symlink of the original.
        if isinstance(code2, dict):
            target_filename = next(iter(code2.keys()))
        else:
            target_filename = f"main.{'cu' if api2 == 'cuda' else 'cpp'}"

        all_examples.append(
            {
                "kernel_name": name,
                "from_api": api1,
                "to_api": api2,
                "source_code": source_code,
                "target_filename": target_filename,
            }
        )

    # Shuffle with a fixed seed for reproducibility, then slice
    import random as _random
    rng = _random.Random(shuffle_seed)
    rng.shuffle(all_examples)

    total_needed = num_train + num_val
    if len(all_examples) < total_needed:
        logging.warning(
            f"Only {len(all_examples)} compilable train kernels available, "
            f"requested {total_needed}. Reducing."
        )
        total_needed = len(all_examples)
        num_train = int(total_needed * 0.6)
        num_val = total_needed - num_train

    trainset = all_examples[:num_train]
    valset = all_examples[num_train : num_train + num_val]

    logging.info(
        f"Loaded {len(trainset)} train + {len(valset)} val examples "
        f"from train.jsonl (pool size: {len(all_examples)}); "
        f"test.jsonl untouched."
    )
    return trainset, valset


# Entry point
def parse_args():
    p = argparse.ArgumentParser(description="GEPA prompt optimization for UniPar_AI")
    p.add_argument(
        "--dataset_dir",
        default=None,
        help="Path to HeCBench dataset directory containing test.jsonl",
    )
    p.add_argument(
        "--hecbench_omp_dir",
        default=None,
        help="Path to hecbench_omp directory (contains src/kernel-api/ with Makefiles)",
    )
    p.add_argument("--from_api", default="omp", choices=["serial", "omp", "cuda", "hip", "sycl"])
    p.add_argument("--to_api", default="cuda", choices=["omp", "cuda"])
    p.add_argument("--num_train", type=int, default=30, help="Number of training kernels")
    p.add_argument("--num_val", type=int, default=20, help="Number of validation kernels")
    p.add_argument("--max_metric_calls", type=int, default=100, help="GEPA evaluation budget")
    p.add_argument(
        "--reflection_lm",
        default="openai/gpt-4o",
        help="LiteLLM model string for GEPA reflection (e.g. openai/gpt-4o, anthropic/claude-sonnet-4-6)",
    )
    p.add_argument(
        "--reflection_base_url",
        default=None,
        help="Base URL for local reflection LM (e.g. http://localhost:8006/v1). "
             "When set, overrides the default LiteLLM routing so a local vLLM server is used.",
    )
    p.add_argument(
        "--reflection_api_key",
        default=None,
        help="API key for the reflection LM server (default: uses OPENAI_API_KEY env var).",
    )
    p.add_argument(
        "--translator_url",
        default="http://localhost:8004/v1",
        help="vLLM server base URL for the translation model",
    )
    p.add_argument(
        "--translator_model",
        default="Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
        help="Model name served by vLLM",
    )
    p.add_argument(
        "--translator_api_key",
        default="token123",
        help="API key for the translation vLLM server (default: token123)",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save GEPA run logs and optimized candidate (default: output/gepa_<timestamp>)",
    )
    p.add_argument("--no_parallel", action="store_true", help="Disable parallel kernel evaluation")
    p.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Max tokens for translation/fix model output (default: 4096; keep below model context limit minus prompt size)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent

    dataset_dir = args.dataset_dir or str(project_root / "data" / "Datasets" / "HeCBench")
    hecbench_omp_dir = args.hecbench_omp_dir or str(project_root / "data" / "Datasets" / "HeCBench")
    output_dir = args.output_dir or str(
        project_root / "output" / f"gepa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Dataset dir:      {dataset_dir}")
    logging.info(f"HeCBench OMP dir: {hecbench_omp_dir}")
    logging.info(f"Output dir:       {output_dir}")

    # Load dataset
    trainset, valset = load_examples(
        dataset_dir=dataset_dir,
        from_api=args.from_api,
        to_api=args.to_api,
        num_train=args.num_train,
        num_val=args.num_val,
        hecbench_omp_dir=hecbench_omp_dir,
    )

    # Initialize model agent
    model_agent = ModelAgent(
        translator_base_url=args.translator_url,
        translator_model=args.translator_model,
        fixer_base_url=args.translator_url,
        fixer_model=args.translator_model,
        api_key=args.translator_api_key,
        max_tokens=args.max_tokens,
    )

    # Build evaluator 
    evaluator = make_evaluator(model_agent=model_agent, hecbench_omp_dir=hecbench_omp_dir,
                               max_metric_calls=args.max_metric_calls)

    # Build reflection LM
    # When a local base URL is given, construct the LiteLLM callable directly so
    # we can pass base_url and api_key without relying on environment variables.
    if args.reflection_base_url:
        from gepa.optimize_anything import make_litellm_lm
        reflection_lm = make_litellm_lm(
            args.reflection_lm,
            base_url=args.reflection_base_url,
            api_key=args.reflection_api_key or "token123",
        )
    else:
        reflection_lm = args.reflection_lm  # LiteLLM resolves via env vars (cloud APIs)

    # Run GEPA
    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=args.max_metric_calls,
            parallel=not args.no_parallel,
            # Keep 2 workers so vLLM isn't overwhelmed; increase if you have capacity
            max_workers=2,
            run_dir=output_dir,
            display_progress_bar=False,
        ),
        reflection=ReflectionConfig(
            reflection_lm=reflection_lm,
            reflection_minibatch_size=3,
        ),
    )

    logging.info("Starting GEPA optimization...")
    result = optimize_anything(
        seed_candidate=SEED_CANDIDATE,
        evaluator=evaluator,
        dataset=trainset,
        valset=valset,
        objective=(
            "Optimize the system_prompt and user_instruction for translating HPC kernel code "
            f"from {args.from_api} to {args.to_api}. "
            "The goal is to produce translated code that compiles correctly on the first attempt, "
            "without needing model-assisted compilation fixes. "
            "Higher score = fewer compilation errors = better translation quality."
        ),
        background=(
            "The kernels are from HeCBench, a suite of HPC benchmarks. "
            f"Source API: {args.from_api.upper()}. Target API: {args.to_api.upper()}. "
            "Common failure patterns: incorrect memory indexing, wrong API function names, "
            "missing headers, invalid thread/block dimension syntax, "
            "incorrect pragma directives or kernel launch syntax. "
            f"The translation model is a local code LLM ({args.translator_model}). "
            "Prompts should be precise, unambiguous, and guide the model to produce "
            "complete, syntactically correct code without placeholder comments."
        ),
        config=config,
    )

    # Save results
    best_candidate = result.best_candidate
    best_score = result.val_aggregate_scores[result.best_idx]

    logging.info(f"Optimization complete. Best val score: {best_score:.3f}")
    if isinstance(best_candidate, dict):
        logging.info("Optimized system_prompt:\n%s", best_candidate.get("system_prompt", ""))
        logging.info("Optimized user_instruction:\n%s", best_candidate.get("user_instruction", ""))
    else:
        logging.info("Optimized candidate:\n%s", best_candidate)

    output_path = os.path.join(output_dir, "optimized_candidate.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "best_val_score": best_score,
                "best_candidate": best_candidate,
                "seed_candidate": SEED_CANDIDATE,
                "config": {
                    "from_api": args.from_api,
                    "to_api": args.to_api,
                    "num_train": args.num_train,
                    "num_val": args.num_val,
                    "max_metric_calls": args.max_metric_calls,
                    "reflection_lm": args.reflection_lm,
                },
            },
            f,
            indent=2,
        )
    logging.info(f"Saved optimized candidate to {output_path}")
    return best_candidate


if __name__ == "__main__":
    main()
