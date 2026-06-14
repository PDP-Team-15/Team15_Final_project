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
#include <random>
#include <chrono>
#include <cuda_runtime.h>

#define VERTICES 600
#define BLOCK_SIZE_X 256

typedef struct __attribute__((__aligned__(8))) {
  float x, y;
} float2;

#include "kernel.h"

void check_cuda_error(cudaError_t error) {
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(EXIT_FAILURE);
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

  std::default_random_engine rng(123);
  std::normal_distribution<float> distribution(0, 1);

  float2* point;
  check_cuda_error(cudaMalloc((void**)&point, sizeof(float2) * nPoints));
  for (int i = 0; i < nPoints; i++) {
    point[i].x = distribution(rng);
    point[i].y = distribution(rng);
  }

  float2* vertex;
  check_cuda_error(cudaMalloc((void**)&vertex, vertices * sizeof(float2)));
  for (int i = 0; i < vertices; i++) {
    float t = distribution(rng) * 2.f * M_PI;
    vertex[i].x = cosf(t);
    vertex[i].y = sinf(t);
  }

  int* bitmap_ref;
  int* bitmap_opt;
  check_cuda_error(cudaMalloc((void**)&bitmap_ref, nPoints * sizeof(int)));
  check_cuda_error(cudaMalloc((void**)&bitmap_opt, nPoints * sizeof(int)));

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    pnpoly_base<<<(nPoints + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, BLOCK_SIZE_X>>>(
        bitmap_ref, point, vertex, nPoints);
    check_cuda_error(cudaGetLastError());
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_base): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    pnpoly_opt_1<<<(nPoints + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, BLOCK_SIZE_X>>>(
        bitmap_opt, point, vertex, nPoints);
    check_cuda_error(cudaGetLastError());
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<1>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    pnpoly_opt_2<<<(nPoints + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, BLOCK_SIZE_X>>>(
        bitmap_opt, point, vertex, nPoints);
    check_cuda_error(cudaGetLastError());
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<2>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    pnpoly_opt_4<<<(nPoints + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, BLOCK_SIZE_X>>>(
        bitmap_opt, point, vertex, nPoints);
    check_cuda_error(cudaGetLastError());
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<4>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    pnpoly_opt_8<<<(nPoints + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, BLOCK_SIZE_X>>>(
        bitmap_opt, point, vertex, nPoints);
    check_cuda_error(cudaGetLastError());
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<8>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    pnpoly_opt_16<<<(nPoints + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, BLOCK_SIZE_X>>>(
        bitmap_opt, point, vertex, nPoints);
    check_cuda_error(cudaGetLastError());
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<16>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    pnpoly_opt_32<<<(nPoints + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, BLOCK_SIZE_X>>>(
        bitmap_opt, point, vertex, nPoints);
    check_cuda_error(cudaGetLastError());
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<32>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    pnpoly_opt_64<<<(nPoints + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, BLOCK_SIZE_X>>>(
        bitmap_opt, point, vertex, nPoints);
    check_cuda_error(cudaGetLastError());
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<64>): %f (s)\n", (time * 1e-9f) / repeat);

  check_cuda_error(cudaMemcpy(bitmap_ref, bitmap_ref, nPoints * sizeof(int), cudaMemcpyDeviceToHost));
  check_cuda_error(cudaMemcpy(bitmap_opt, bitmap_opt, nPoints * sizeof(int), cudaMemcpyDeviceToHost));

  int error = memcmp(bitmap_opt, bitmap_ref, nPoints * sizeof(int));

  int checksum = 0;
  for (int i = 0; i < nPoints; i++) checksum += bitmap_opt[i];
  printf("Checksum: %d\n", checksum);

  printf("%s\n", error ? "FAIL" : "PASS");

  check_cuda_error(cudaFree(point));
  check_cuda_error(cudaFree(vertex));
  check_cuda_error(cudaFree(bitmap_ref));
  check_cuda_error(cudaFree(bitmap_opt));

  return 0;
}