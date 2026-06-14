```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <random>
#include <chrono>

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

  float2 *point;
  cudaMalloc((void**)&point, nPoints * sizeof(float2));
  for (int i = 0; i < nPoints; i++) {
    point[i].x = distribution(rng);
    point[i].y = distribution(rng);
  }

  float2 *vertex;
  cudaMalloc((void**)&vertex, vertices * sizeof(float2));
  for (int i = 0; i < vertices; i++) {
    float t = distribution(rng) * 2.f * M_PI;
    vertex[i].x = cosf(t);
    vertex[i].y = sinf(t);
  }

  int *bitmap_ref;
  int *bitmap_opt;
  cudaMalloc((void**)&bitmap_ref, nPoints * sizeof(int));
  cudaMalloc((void**)&bitmap_opt, nPoints * sizeof(int));

  cudaMemset(bitmap_ref, 0, nPoints * sizeof(int));
  cudaMemset(bitmap_opt, 0, nPoints * sizeof(int));

  dim3 blockSize(BLOCK_SIZE_X, 1, 1);
  dim3 gridSize((nPoints + blockSize.x - 1) / blockSize.x, 1, 1);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    pnpoly_base<<<gridSize, blockSize>>>(bitmap_ref, point, vertex, nPoints);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_base): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    pnpoly_opt<1><<<gridSize, blockSize>>>(bitmap_opt, point, vertex, nPoints);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<1>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    pnpoly_opt<2><<<gridSize, blockSize>>>(bitmap_opt, point, vertex, nPoints);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<2>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    pnpoly_opt<4><<<gridSize, blockSize>>>(bitmap_opt, point, vertex, nPoints);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<4>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    pnpoly_opt<8><<<gridSize, blockSize>>>(bitmap_opt, point, vertex, nPoints);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (