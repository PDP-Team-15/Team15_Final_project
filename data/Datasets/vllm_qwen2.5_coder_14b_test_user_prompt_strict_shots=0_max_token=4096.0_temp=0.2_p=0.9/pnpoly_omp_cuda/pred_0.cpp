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

__global__ void pnpoly_base_kernel(int* bitmap, float2* point, float2* vertex, int nPoints, int vertices) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nPoints) return;

  int inside = 0;
  for (int i = 0, j = vertices - 1; i < vertices; j = i++) {
    if (((vertex[i].y > point[idx].y) != (vertex[j].y > point[idx].y)) &&
        (point[idx].x < (vertex[j].x - vertex[i].x) * (point[idx].y - vertex[i].y) / (vertex[j].y - vertex[i].y) + vertex[i].x)) {
      inside = !inside;
    }
  }
  bitmap[idx] = inside;
}

__global__ void pnpoly_opt_kernel(int* bitmap, float2* point, float2* vertex, int nPoints, int vertices, int stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nPoints) return;

  int inside = 0;
  for (int i = 0, j = vertices - 1; i < vertices; j = i++) {
    if (((vertex[i].y > point[idx].y) != (vertex[j].y > point[idx].y)) &&
        (point[idx].x < (vertex[j].x - vertex[i].x) * (point[idx].y - vertex[i].y) / (vertex[j].y - vertex[i].y) + vertex[i].x)) {
      inside = !inside;
    }
  }
  bitmap[idx] = inside;
}

void pnpoly_base(int* bitmap, float2* point, float2* vertex, int nPoints) {
  int threadsPerBlock = BLOCK_SIZE_X;
  int blocksPerGrid = (nPoints + threadsPerBlock - 1) / threadsPerBlock;

  pnpoly_base_kernel<<<blocksPerGrid, threadsPerBlock>>>(bitmap, point, vertex, nPoints, VERTICES);
  cudaDeviceSynchronize();
}

template<int stride>
void pnpoly_opt(int* bitmap, float2* point, float2* vertex, int nPoints) {
  int threadsPerBlock = BLOCK_SIZE_X;
  int blocksPerGrid = (nPoints + threadsPerBlock - 1) / threadsPerBlock;

  pnpoly_opt_kernel<<<blocksPerGrid, threadsPerBlock>>>(bitmap, point, vertex, nPoints, VERTICES, stride);
  cudaDeviceSynchronize();
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

  float2 *point, *vertex;
  int *bitmap_ref, *bitmap_opt;

  cudaMallocManaged(&point, sizeof(float2) * nPoints);
  cudaMallocManaged(&vertex, vertices * sizeof(float2));
  cudaMallocManaged(&bitmap_ref, nPoints * sizeof(int));
  cudaMallocManaged(&bitmap_opt, nPoints * sizeof(int));

  for (int i = 0; i < nPoints; i++) {
    point[i].x = distribution(rng);
    point[i].y = distribution(rng);
  }

  for (int i = 0; i < vertices; i++) {
    float t = distribution(rng) * 2.f * M_PI;
    vertex[i].x = cosf(t);
    vertex[i].y = sinf(t);
  }

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

  int error = memcmp(bitmap_opt, bitmap_ref, nPoints*sizeof(int)); 

  int checksum = 0;
  for (int i = 0; i < nPoints; i++) checksum += bitmap_opt[i];
  printf("Checksum: %d\n", checksum);

  printf("%s\n", error ? "FAIL" : "PASS");

  cudaFree(vertex);
  cudaFree(point);
  cudaFree(bitmap_ref);
  cudaFree(bitmap_opt);
  return 0;
}
```