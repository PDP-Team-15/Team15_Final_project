#run this in terminal before running script: export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

import os

# Both sides must have exactly 1 file (18 kernels each direction)
CANONICAL_KERNELS = {
    "omp-cuda": {'ace', 'convolution1D', 'geodesic', 'heat', 'keogh', 'meanshift',
                 'michalewicz', 'mixbench', 'particlefilter', 'pnpoly', 'pool',
                 'stddev', 'stencil1d', 'su3', 'vanGenuchten', 'winograd', 'wsm5',
                 'zmddft'},
    "cuda-omp": {'ace', 'convolution1D', 'geodesic', 'heat', 'keogh', 'meanshift',
                 'michalewicz', 'mixbench', 'particlefilter', 'pnpoly', 'pool',
                 'stddev', 'stencil1d', 'su3', 'vanGenuchten', 'winograd', 'wsm5',
                 'zmddft'},
}
import subprocess
import time
import json
import sys
import multiprocessing
import argparse
from datetime import datetime


class Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def process_kernel(params):
    """Process a single kernel: compile then run separately."""
    kernel, benchmark_path, to_api = params
    code_path = os.path.join(benchmark_path, 'src')
    curr_dir_path = os.path.join(code_path, kernel)

    if not os.path.isdir(curr_dir_path):
        return 0, 0, 1, kernel, None  # (compiled, correct, total, kernel, time)

    os.chdir(curr_dir_path)
    try:
        subprocess.run(['make', 'clean'], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Make clean failed for {kernel}: {e.stderr}")

    # Step 1: compile only
    try:
        if to_api == 'omp':
            subprocess.run(['make', '-f', 'Makefile.aomp', 'DEVICE=cpu', '-j'],
                           check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            subprocess.run(['make', '-j'],
                           check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"\n**Compile error {kernel}: {e.stderr}\n")
        return 0, 0, 1, kernel, None  # compile failed

    # Step 2: run and check correctness
    try:
        timeout_time = 620
        start = time.time()
        if to_api == 'omp':
            res = subprocess.run(['make', 'run', '-f', 'Makefile.aomp', '-j', 'DEVICE=cpu'],
                                 check=True, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, timeout=timeout_time)
        else:
            res = subprocess.run(['make', 'run', '-j'],
                                 check=True, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, timeout=timeout_time)
    except subprocess.CalledProcessError as e:
        print(f"\n**Run error {kernel}: {e.stderr}\n")
        return 1, 0, 1, kernel, None  # compiled but run failed
    except subprocess.TimeoutExpired:
        print(f"\n**Timeout expired for {kernel}: {benchmark_path}\n")
        return 1, 0, 1, kernel, None  # compiled but timed out

    elapsed = time.time() - start
    print(f"\n**Execution out {kernel}: {res.stdout.strip()}\n")
    correct = 1 if 'pass' in res.stdout.lower() else 0
    return 1, correct, 1, kernel, elapsed if correct else None


def eval_run(benchmark_path, to_api, canonical_kernels=None):
    code_path = os.path.join(benchmark_path, 'src')

    if not os.path.exists(code_path):
        print(f"Path does not exist: {code_path}")
        if canonical_kernels:
            print(f"  [canonical] all {len(canonical_kernels)} kernels counted as fail: {sorted(canonical_kernels)}")
            return 0, 0, len(canonical_kernels), {}
        return 0, 0, 0, {}

    try:
        kernels = [k for k in os.listdir(code_path) if os.path.isdir(os.path.join(code_path, k))]
    except FileNotFoundError:
        print(f"Directory not found: {code_path}")
        return 0, 0, 0, {}

    if canonical_kernels:
        # Only evaluate kernels in the canonical set; ignore spurious dirs (e.g. nested 'src')
        kernels = [k for k in kernels if k.rsplit('-', 1)[0] in canonical_kernels]
        present_names = {k.rsplit('-', 1)[0] for k in kernels}
        missing = sorted(canonical_kernels - present_names)
    else:
        missing = []

    kernel_params = [(kernel, benchmark_path, to_api) for kernel in kernels]
    num_pool = 1 if to_api == 'omp' else 4
    with multiprocessing.Pool(processes=min(len(kernel_params), num_pool)) as pool:
        results = pool.map(process_kernel, kernel_params)

    for m in missing:
        print(f"  [canonical] missing kernel counted as fail: {m}")
        results.append((0, 0, 1, m, None))

    num_compiled = sum(r[0] for r in results)
    num_correct  = sum(r[1] for r in results)
    total        = sum(r[2] for r in results)
    execution_times = {r[3]: r[4] for r in results if r[4] is not None}
    return num_compiled, num_correct, total, execution_times


if __name__ == '__main__':
    HOME_PATH = os.path.expanduser('~')
    parser = argparse.ArgumentParser(description="Run kernel benchmarks for a target API")
    parser.add_argument("--target", choices=["cuda", "omp"], nargs='+', required=True, help="Target API(s) to evaluate")
    parser.add_argument('--from_apis', nargs='+', default=['cuda', 'omp', 'serial'], help='Source API(s) to include (default: cuda omp serial)')
    parser.add_argument('--run_names', nargs='+', required=True, help='List of run names to evaluate')
    parser.add_argument('--log', type=str, default=None, help='Path to log file (default: auto-generated in eval dir)')

    args = parser.parse_args()
    run_names_to_process = args.run_names
    target_apis = args.target
    from_apis = args.from_apis

    # Set up log file (tee stdout to file)
    timestamp = datetime.now().strftime("%d-%m-%y_%H-%M")
    target_str = "_".join(target_apis)
    log_path = args.log if args.log else f"eval_{target_str}_{timestamp}.log"
    sys.stdout = Tee(log_path)

    print("Selected target APIs:", target_apis)
    print("Selected from APIs:", from_apis)

    for run_name in run_names_to_process:
        base_candidates = [
            os.path.join(HOME_PATH, f'UniPar_AI/Datasets/eval/{run_name}'),
            os.path.join(HOME_PATH, f'unipar/Datasets/eval/{run_name}')
        ]
        BASE = next((path for path in base_candidates if os.path.isdir(path)), base_candidates[0])
        print(BASE)
        try:
            the_dir = os.listdir(BASE)
        except FileNotFoundError:
            print(f"Skipping {BASE} - directory does not exist")
            continue

        results = []
        per_pair_results = {}
        compiled_cuda, correct_cuda, total_cuda = 0, 0, 0
        compiled_omp,  correct_omp,  total_omp  = 0, 0, 0
        all_times = {}

        for translation in the_dir:
            if len(translation.rsplit('-')) < 3:
                print("skipping: ", translation)
                continue
            from_api = translation.rsplit('-')[1]
            to_api   = translation.rsplit('-')[2]
            if from_api == to_api:
                continue
            if (to_api not in target_apis) or (from_api not in from_apis):
                continue

            pair_key = f"{from_api}-{to_api}"
            print("\n********Translate total for: ", translation, "********\n")
            canonical = CANONICAL_KERNELS.get(pair_key)
            num_compiled, num_correct, total, times = eval_run(os.path.join(BASE, translation), to_api, canonical_kernels=canonical)
            all_times[f"{from_api}-{to_api}"] = times
            results.append((num_compiled, num_correct, total))
            per_pair_results[pair_key] = {
                "compile": f"{num_compiled}/{total}",
                "compile_rate": round(num_compiled / total, 3) if total > 0 else None,
                "correct": f"{num_correct}/{total}",
                "correct_rate": round(num_correct / total, 3) if total > 0 else None,
            }

            if total == 0:
                print(from_api, "-", to_api, "compile rate: n/a  correct rate: n/a")
            else:
                print(from_api, "-", to_api,
                      f"compile rate: {num_compiled}/{total} ({num_compiled/total:.3f})",
                      f"| correct rate: {num_correct}/{total} ({num_correct/total:.3f})")

            if to_api == 'cuda':
                compiled_cuda += num_compiled
                correct_cuda  += num_correct
                total_cuda    += total
            elif to_api == 'omp':
                compiled_omp += num_compiled
                correct_omp  += num_correct
                total_omp    += total

        with open(os.path.join(BASE, 'execution_times.json'), 'w') as f:
            json.dump(all_times, f, indent=4)

        print()
        if total_cuda > 0:
            print(f"cuda compile rate: {compiled_cuda}/{total_cuda} ({compiled_cuda/total_cuda:.3f})")
            print(f"cuda correct rate: {correct_cuda}/{total_cuda} ({correct_cuda/total_cuda:.3f})")
        if total_omp > 0:
            print(f"omp compile rate: {compiled_omp}/{total_omp} ({compiled_omp/total_omp:.3f})")
            print(f"omp correct rate: {correct_omp}/{total_omp} ({correct_omp/total_omp:.3f})")

        print()
        for num_compiled, num_correct, total in results:
            if total == 0:
                print("=ROUND(0/0,3)", end="\t")
            else:
                print(f"compile={num_compiled}/{total} correct={num_correct}/{total}", end="\t")
        print()
        sys.stdout.flush()

        # Save structured summary JSON
        summary = {
            "run_name": run_name,
            "target": target_apis,
            "per_pair": per_pair_results,
            "aggregate": {}
        }
        if total_cuda > 0:
            summary["aggregate"]["cuda"] = {
                "compile": f"{compiled_cuda}/{total_cuda}",
                "compile_rate": round(compiled_cuda / total_cuda, 3),
                "correct": f"{correct_cuda}/{total_cuda}",
                "correct_rate": round(correct_cuda / total_cuda, 3),
            }
        if total_omp > 0:
            summary["aggregate"]["omp"] = {
                "compile": f"{compiled_omp}/{total_omp}",
                "compile_rate": round(compiled_omp / total_omp, 3),
                "correct": f"{correct_omp}/{total_omp}",
                "correct_rate": round(correct_omp / total_omp, 3),
            }
        summary_path = os.path.join(BASE, 'eval_results.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"\nResults saved to: {summary_path}")
        if isinstance(sys.stdout, Tee):
            print(f"Log saved to: {log_path}")

    if isinstance(sys.stdout, Tee):
        tee = sys.stdout
        sys.stdout = tee.terminal
        tee.log.close()
