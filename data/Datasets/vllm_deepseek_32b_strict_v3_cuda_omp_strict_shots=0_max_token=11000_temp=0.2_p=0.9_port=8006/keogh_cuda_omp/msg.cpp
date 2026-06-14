You are an HPC expert specializing in translating CUDA code to OpenMP. Your goal is to produce syntactically correct OpenMP code that compiles on the first attempt without any errors. Follow these guidelines:
1. Remove all CUDA-specific constructs and qualifiers such as `__global__`, `__device__`, `__host__`, `__forceinline__`, and any CUDA-specific type qualifiers.
2. Replace CUDA memory management functions (`cudaMalloc`, `cudaFree`, `cudaMemcpy`) with standard C functions (`malloc`, `free`).
3. Replace CUDA kernel launches with direct function calls or OpenMP parallel regions using `#pragma omp parallel for`.
4. Replace CUDA-specific variables (`blockIdx`, `blockDim`, `threadIdx`) with OpenMP thread management functions like omp_get_num_threads() and `omp_get_thread_num()`.
5. Remove any CUDA-specific includes like #include <cuda.h> and ensure only necessary OpenMP includes like #include <omp.h> are present.
6. Ensure all function parameters and return types are compatible with OpenMP and remove any unused parameters.
7. Check and ensure all variables and function signatures match exactly between CUDA source and OpenMP implementation.
8. Use standard C functions for any mathematical or utility operations that do not require OpenMP-specific handling.
9. Output only the translated code without any comments, explanations, or debugging statements.
For each CUDA kernel code provided, systematically and accurately translate it to OpenMP. Include all necessary OpenMP headers and ensure correct function replacements. Remove any CUDA-specific code. Output a complete, functional OpenMP implementation ready for direct compilation without errors. Pay special attention to proper use of OpenMP functions, correct thread and task constructs, and appropriate memory management. Do not retain any CUDA-specific function calls or variables. Include all required headers at the beginning of the code. Ensure the resulting code is syntactically correct, properly formatted, and compiles without errors on the first attempt.
Translate the following code  from cuda to omp:
main.cu:
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <cuda.h>
#include "reference.h"



__global__
void lb_keogh(const float *__restrict__ subject,
              const float *__restrict__ avgs,
              const float *__restrict__ stds, 
                    float *__restrict__ lb_keogh,
              const float *__restrict__ lower_bound,
              const float *__restrict__ upper_bound,
              const int M,
              const int N) 
{
  

  extern __shared__ float cache[];

  int lid = threadIdx.x;
  int blockSize = blockDim.x * blockIdx.x;
  int idx = blockSize + lid;

  for (int k = lid; k < blockDim.x + M; k += blockDim.x)
    if (blockSize + k < N) cache[k] = subject[blockSize + k];

  __syncthreads();

  if (idx < N-M+1) {

    

    float residues = 0;
    float avg = avgs[idx];
    float std = stds[idx];

    for (int i = 0; i < M; ++i) {
      

      float value = (cache[lid+i] - avg) / std;
      float lower = value - lower_bound[i];
      float upper = value - upper_bound[i];

      

      residues += upper*upper*(upper > 0) + lower*lower*(lower < 0);
    }

    lb_keogh[idx] = residues;
  }
}

int main(int argc, char* argv[]) {

  if (argc != 4) {
    printf("Usage: ./%s <query length> <subject length> <repeat>\n", argv[0]);
    return 1;
  }

  const int M = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  printf("Query length = %d\n", M);
  printf("Subject length = %d\n", N);

  

  float *subject = (float*) malloc (sizeof(float)*N);
  float *lower = (float*) malloc (sizeof(float)*N);
  float *upper = (float*) malloc (sizeof(float)*N);
  float *lb = (float*) malloc (sizeof(float)*(N-M+1));
  float *lb_h = (float*) malloc (sizeof(float)*(N-M+1));
  float *avgs = (float*) malloc (sizeof(float)*(N-M+1));
  float *stds = (float*) malloc (sizeof(float)*(N-M+1));

  srand(123);
  for (int i = 0; i < N; ++i) subject[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < N-M+1; ++i) avgs[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < N-M+1; ++i) stds[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < M; ++i) upper[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < M; ++i) lower[i] = (float)rand() / (float)RAND_MAX;

  float *d_subject = NULL, *d_avgs = NULL, *d_stds = NULL, 
        *d_lb = NULL, *d_lower = NULL, *d_upper = NULL;

  cudaMalloc(&d_subject, sizeof(float)*N);
  cudaMalloc(&d_avgs, sizeof(float)*(N-M+1));
  cudaMalloc(&d_stds, sizeof(float)*(N-M+1));
  cudaMalloc(&d_lb, sizeof(float)*(N-M+1));
  cudaMalloc(&d_lower, sizeof(float)*M);
  cudaMalloc(&d_upper, sizeof(float)*M);

  cudaMemcpy(d_subject, subject, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_avgs, avgs, sizeof(float)*(N-M+1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_stds, stds, sizeof(float)*(N-M+1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lower, lower, sizeof(float)*M, cudaMemcpyHostToDevice);
  cudaMemcpy(d_upper, upper, sizeof(float)*M, cudaMemcpyHostToDevice);

  const int blocks = 256;
  const int grids = (N-M+1 + blocks - 1) / blocks;
  int smem_size = (M+blocks)*sizeof(float);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    lb_keogh<<<grids, blocks, smem_size>>>
      (d_subject, d_avgs, d_stds, d_lb, d_lower, d_upper, M, N);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  cudaMemcpy(lb, d_lb, sizeof(float)*(N-M+1), cudaMemcpyDeviceToHost);

  

  reference(subject, avgs, stds, lb_h, lower, upper, M, N);
  bool ok = true;
  for (int i = 0; i < N-M+1; i++) {
    if (fabsf(lb[i] - lb_h[i]) > 1e-3f) {
      printf("%d %f %f\n", i, lb[i], lb_h[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  cudaFree(d_lb);
  cudaFree(d_avgs);
  cudaFree(d_stds);
  cudaFree(d_subject);
  cudaFree(d_lower);
  cudaFree(d_upper);
  free(lb);
  free(lb_h);
  free(avgs);
  free(stds);
  free(subject);
  free(lower);
  free(upper);
  return 0;
}
