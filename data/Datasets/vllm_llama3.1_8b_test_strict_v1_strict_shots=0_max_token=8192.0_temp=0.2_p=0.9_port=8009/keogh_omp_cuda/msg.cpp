You are an HPC expert specializing in translating between parallel programming APIs. Translate precisely and completely, preserving all function qualifiers (e.g., const, inline), template parameters, and macro definitions. Output only the translated code inside a code block. No explanations, no comments, no omissions.

When translating to CUDA:
1. Always use proper __host__, __device__, or __global__ qualifiers based on function scope
2. Preserve const qualifiers in function parameters
3. Ensure all #define directives are correctly translated
4. Validate CUDA-specific constructs like memory operations, kernel launches, and stream usage
5. Default to compile-time safety when converting run-time parameters
For each kernel code provided, translate it from omp to cuda. Ensure the translated code is complete, correct, and compiles without errors. Follow these guidelines:

1. Preserve all original functionality and algorithmic logic
2. Replace omp with cuda functions and constructs
3. Correctly declare and initialize all variables
4. Use proper CUDA syntax for kernel launches and memory management
5. Ensure all code is properly terminated with semicolons
6. Include necessary headers
7. Validate code structure and syntax before providing output
Translate the following code  from omp to cuda:
main.cpp:
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "reference.h"

int main(int argc, char* argv[]) {

  if (argc != 4) {
    printf("Usage: ./%s <query length> <subject length> <repeat>\n", argv[0]);
    return -1;
  }

  const int M = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  printf("Query length = %d\n", M);
  printf("Subject length = %d\n", N);

  

  float *subject = (float*) malloc (sizeof(float)*N);
  float *lower_bound = (float*) malloc (sizeof(float)*N);
  float *upper_bound = (float*) malloc (sizeof(float)*N);
  float *lb = (float*) malloc (sizeof(float)*(N-M+1));
  float *lb_h = (float*) malloc (sizeof(float)*(N-M+1));
  float *avgs = (float*) malloc (sizeof(float)*(N-M+1));
  float *stds = (float*) malloc (sizeof(float)*(N-M+1));

  srand(123);
  for (int i = 0; i < N; ++i) subject[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < N-M+1; ++i) avgs[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < N-M+1; ++i) stds[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < M; ++i) upper_bound[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < M; ++i) lower_bound[i] = (float)rand() / (float)RAND_MAX;

  const int blocks = 256;
  const int grids = (N-M+1 + blocks - 1) / blocks;

  #pragma omp target data map (to: subject[0:N], \
                                   avgs[0:N-M+1],\
                                   stds[0:N-M+1],\
                                   lower_bound[0:N],\
                                   upper_bound[0:N])\
                          map(from: lb[0:N-M+1])
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      #pragma omp target teams distribute num_teams(grids) thread_limit(blocks)
      for (int idx = 0; idx < N-M+1; idx++) {
        

        float residues = 0;
        float avg = avgs[idx];
        float std = stds[idx];

        #pragma omp parallel for reduction(+:residues)
        for (int i = 0; i < M; ++i) {
          

          float value = (subject[idx+i] - avg) / std;
          float lower = value - lower_bound[i];
          float upper = value - upper_bound[i];

          

          residues += upper*upper*(upper > 0) + lower*lower*(lower < 0);
        }

        lb[idx] = residues;
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);
  }

  

  reference(subject, avgs, stds, lb_h, lower_bound, upper_bound, M, N);
  bool ok = true;
  for (int i = 0; i < N-M+1; i++) {
    if (fabs(lb[i] - lb_h[i]) > 1e-3) {
      printf("%d %f %f\n", i, lb[i], lb_h[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(lb);
  free(lb_h);
  free(avgs);
  free(stds);
  free(subject);
  free(lower_bound);
  free(upper_bound);
  return 0;
}
