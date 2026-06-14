#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <chrono>
#include <omp.h>
#include "kernel.h"
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
  float2 *vertex;
  cudaMallocHost((void**)&point, nPoints * sizeof(float2));
  cudaMallocHost((void**)&vertex, vertices * sizeof(float2));

  for (int i = 0; i < nPoints; i++) {
    point[i].x = distribution(rng);
    point[i].y = distribution(rng);
  }

  for (int i = 0; i < vertices; i++) {
    float t = distribution(rng) * 2.f * M_PI;
    vertex[i].x = cosf(t);
    vertex[i].y = sinf(t);
  }

  int *bitmap_ref;
  int *bitmap_opt;
  cudaMallocHost((void**)&bitmap_ref, nPoints * sizeof(int));
  cudaMallocHost((void**)&bitmap_opt, nPoints * sizeof(int));

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  for (int i = 0; i < repeat; i++) {
    cudaEventRecord(start, 0);
    pnpoly_base<<<1, 1>>>(bitmap_ref, point, vertex, nPoints);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    printf("Average kernel execution time (pnpoly_base): %f (s)\n", time / 1000.0f / repeat);
  }

  for (int i = 0; i < repeat; i++) {
    cudaEventRecord(start, 0);
    pnpoly_opt<1> <<<1, 1>>> (bitmap_opt, point, vertex, nPoints);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    printf("Average kernel execution time (pnpoly_opt<1>): %f (s)\n", time / 1000.0f / repeat);
  }

  for (int i = 0; i < repeat; i++) {
    cudaEventRecord(start, 0);
    pnpoly_opt<2> <<<1, 1>>> (bitmap_opt, point, vertex, nPoints);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    printf("Average kernel execution time (pnpoly_opt<2>): %f (s)\n", time / 1000.0f / repeat);
  }

  for (int i = 0; i < repeat; i++) {
    cudaEventRecord(start, 0);
    pnpoly_opt<4> <<<1, 1>>> (bitmap_opt, point, vertex, nPoints);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    printf("Average kernel execution time (pnpoly_opt<4>): %f (s)\n", time / 1000.0f / repeat);
  }

  for (int i = 0; i < repeat; i++) {
    cudaEventRecord(start, 0);
    pnpoly_opt<8> <<<1, 1>>> (bitmap_opt, point, vertex, nPoints);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    printf("Average kernel execution time (pnpoly_opt<8>): %f (s)\n", time / 1000.0f / repeat);
  }

  for (int i = 0; i < repeat; i++) {
    cudaEventRecord(start, 0);
    pnpoly_opt<16> <<<1, 1>>> (bitmap_opt, point, vertex, nPoints);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    printf("Average kernel execution time (pnpoly_opt<16>): %f (s)\n", time / 1000.0f / repeat);
  }

  for (int i = 0; i < repeat; i++) {
    cudaEventRecord(start, 0);
    pnpoly_opt<32> <<<1, 1>>> (bitmap_opt, point, vertex, nPoints);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    printf("Average kernel execution time (pnpoly_opt<32>): %f (s)\n", time / 1000.0f / repeat);
  }

  for (int i = 0; i < repeat; i++) {
    cudaEventRecord(start, 0);
    pnpoly_opt<64> <<<1, 1>>> (bitmap_opt, point, vertex, nPoints);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    printf("Average kernel execution time (pnpoly_opt<64>): %f (s)\n", time / 1000.0f / repeat);
  }

  int error = memcmp(bitmap_opt, bitmap_ref, nPoints*sizeof(int)); 
  
  int checksum = 0;
  for (int i = 0; i < nPoints; i++) checksum += bitmap_opt[i];
  printf("Checksum: %d\n", checksum);

  printf("%s\n", error ? "FAIL" : "PASS");

  cudaFreeHost(point);
  cudaFreeHost(vertex);
  cudaFreeHost(bitmap_ref);
  cudaFreeHost(bitmap_opt);
  return 0;
}