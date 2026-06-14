You are an HPC expert specializing in translating between parallel programming APIs. You will translate OMP code to CUDA code with the following requirements:

1. Include all necessary CUDA headers (e.g., cuda_runtime.h, etc.)
2. Replace OMP function calls with correct CUDA equivalents
3. Use proper CUDA syntax for kernel launches and memory management
4. Ensure all variables are correctly declared and accessible within CUDA context
5. Function names must match the original code
6. Include complete code only - no partial lines or commented code
7. Output must compile without errors on first attempt
8. CUDA kernel functions must be marked with __global__ 
9. Helper functions must be marked with __device__
10. Include all necessary includes and preprocessor directives
11. Ensure proper memory alignment and indexing
12. Include complete error checking for CUDA API calls (if applicable)
13. Use standard CUDA best practices for code structure and style
14. All code must be syntactically complete - no missing semicolons or parenthesis
15. Preserve the original algorithm's logic and functionality

Output only the translated code inside a code block. No explanations, no comments, no omissions. Ensure all CUDA-specific syntax is properly formatted and complete.
For each kernel code provided, translate it from omp to cuda. Provide the complete code in cuda. Do not truncate or use ellipses. Ensure correctness. Do not add any additional comments or explanations.
Translate the following code  from omp to cuda:
main.cpp:
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <chrono>
#include <omp.h>

#define VERTICES 600
#define BLOCK_SIZE_X 256

typedef struct __attribute__((__aligned__(8)))
{
  float x, y;
} float2;

#include "kernel.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: ./%s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  const int nPoints = 2e7;
  const int vertices = VERTICES;

  std::default_random_engine rng (123);
  std::normal_distribution<float> distribution(0, 1);

  float2 *point = (float2*) malloc (sizeof(float2) * nPoints);
  for (int i = 0; i < nPoints; i++) {
    point[i].x = distribution(rng);
    point[i].y = distribution(rng);
  }

  float2 *vertex = (float2*) malloc (vertices * sizeof(float2));
  for (int i = 0; i < vertices; i++) {
    float t = distribution(rng) * 2.f * M_PI;
    vertex[i].x = cosf(t);
    vertex[i].y = sinf(t);
  }

  

  int *bitmap_ref = (int*) malloc (nPoints * sizeof(int));
  int *bitmap_opt = (int*) malloc (nPoints * sizeof(int));

  #pragma omp target data map (to: point[0:nPoints], \
                                  vertex[0:vertices]) \
                         map (from: bitmap_ref[0:nPoints], \
                                    bitmap_opt[0:nPoints])
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_base(bitmap_ref, point, vertex, nPoints);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_base): %f (s)\n", (time * 1e-9f) / repeat);

    

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_opt<1>(bitmap_opt, point, vertex, nPoints);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_opt<1>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_opt<2>(bitmap_opt, point, vertex, nPoints);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_opt<2>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_opt<4>(bitmap_opt, point, vertex, nPoints);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_opt<4>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_opt<8>(bitmap_opt, point, vertex, nPoints);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_opt<8>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_opt<16>(bitmap_opt, point, vertex, nPoints);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_opt<16>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_opt<32>(bitmap_opt, point, vertex, nPoints);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_opt<32>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_opt<64>(bitmap_opt, point, vertex, nPoints);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_opt<64>): %f (s)\n", (time * 1e-9f) / repeat);
  }

  

  int error = memcmp(bitmap_opt, bitmap_ref, nPoints*sizeof(int)); 
  
  

  int checksum = 0;
  for (int i = 0; i < nPoints; i++) checksum += bitmap_opt[i];
  printf("Checksum: %d\n", checksum);

  printf("%s\n", error ? "FAIL" : "PASS");

  free(vertex);
  free(point);
  free(bitmap_ref);
  free(bitmap_opt);
  return 0;
}
