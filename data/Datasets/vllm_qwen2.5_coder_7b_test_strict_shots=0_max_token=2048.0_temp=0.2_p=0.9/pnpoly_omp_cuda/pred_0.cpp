```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

#define VERTICES 600
#define BLOCK_SIZE_X 256

typedef struct __attribute__((__aligned__(8)))
{
  float x, y;
} float2;

#include "kernel.h"

__global__ void pnpoly_opt(int *bitmap_opt, float2 *point, float2 *vertex, int nPoints) {
  // Kernel implementation goes here
}

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

  cudaMemcpyToSymbol(point_d, point, nPoints * sizeof(float2));
  cudaMemcpyToSymbol(vertex_d, vertex, vertices * sizeof(float2));

  for (int i = 0; i < repeat; i++) {
    pnpoly_base<<<(nPoints + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, BLOCK_SIZE_X>>>(bitmap_ref_d, point_d, vertex_d, nPoints);
    pnpoly_opt<<<(nPoints + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, BLOCK_SIZE_X>>>(bitmap_opt_d, point_d, vertex_d, nPoints);
  }

  cudaMemcpy(bitmap_opt, bitmap_opt_d, nPoints * sizeof(int), cudaMemcpyDeviceToHost);

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
```