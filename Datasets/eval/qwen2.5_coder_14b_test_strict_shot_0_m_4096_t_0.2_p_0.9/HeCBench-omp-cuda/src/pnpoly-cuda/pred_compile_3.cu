#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>

#define VERTICES 600
#define BLOCK_SIZE_X 256

typedef struct __attribute__((__aligned__(8)))
{
  float x, y;
} float2;

__global__ void pnpoly_base_kernel(int* bitmap, float2* point, float2* vertex, int nPoints, int vertices) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nPoints) {
    // Implement the logic of pnpoly_base here
  }
}

template<int N>
__global__ void pnpoly_opt_kernel(int* bitmap, float2* point, float2* vertex, int nPoints, int vertices) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nPoints) {
    // Implement the logic of pnpoly_opt<N> here
  }
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

  float2 *d_point, *d_vertex;
  int *d_bitmap_ref, *d_bitmap_opt;
  cudaMalloc(&d_point, sizeof(float2) * nPoints);
  cudaMalloc(&d_vertex, sizeof(float2) * vertices);
  cudaMalloc(&d_bitmap_ref, sizeof(int) * nPoints);
  cudaMalloc(&d_bitmap_opt, sizeof(int) * nPoints);

  cudaMemcpy(d_point, point, sizeof(float2) * nPoints, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vertex, vertex, sizeof(float2) * vertices, cudaMemcpyHostToDevice);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    pnpoly_base_kernel<<<(nPoints + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, BLOCK_SIZE_X>>>(d_bitmap_ref, d_point, d_vertex, nPoints, vertices);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_base): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    pnpoly_opt_kernel<1><<<(nPoints + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, BLOCK_SIZE_X>>>(d_bitmap_opt, d_point, d_vertex, nPoints, vertices);
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<1>): %f (s)\n", (time * 1e-9f) / repeat);

  // Repeat similar blocks for pnpoly_opt<2>, pnpoly_opt<4>, etc.

  cudaMemcpy(bitmap_opt, d_bitmap_opt, sizeof(int) * nPoints, cudaMemcpyDeviceToHost);

  int error = memcmp(bitmap_opt, bitmap_ref, nPoints*sizeof(int)); 

  int checksum = 0;
  for (int i = 0; i < nPoints; i++) checksum += bitmap_opt[i];
  printf("Checksum: %d\n", checksum);

  printf("%s\n", error ? "FAIL" : "PASS");

  free(vertex);
  free(point);
  free(bitmap_ref);
  free(bitmap_opt);
  cudaFree(d_point);
  cudaFree(d_vertex);
  cudaFree(d_bitmap_ref);
  cudaFree(d_bitmap_opt);
  return 0;
}