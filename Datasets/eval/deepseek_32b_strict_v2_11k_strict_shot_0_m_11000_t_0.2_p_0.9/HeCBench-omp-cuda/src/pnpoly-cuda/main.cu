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

  float2 *h_point = (float2*)malloc(sizeof(float2) * nPoints);
  for (int i = 0; i < nPoints; i++) {
    h_point[i].x = distribution(rng);
    h_point[i].y = distribution(rng);
  }

  float2 *h_vertex = (float2*)malloc(vertices * sizeof(float2));
  for (int i = 0; i < vertices; i++) {
    float t = distribution(rng) * 2.f * M_PI;
    h_vertex[i].x = cosf(t);
    h_vertex[i].y = sinf(t);
  }

  float2 *d_point;
  float2 *d_vertex;
  int *d_bitmap_ref;
  int *d_bitmap_opt;

  cudaMalloc((void**)&d_point, sizeof(float2) * nPoints);
  cudaMalloc((void**)&d_vertex, sizeof(float2) * vertices);
  cudaMalloc((void**)&d_bitmap_ref, sizeof(int) * nPoints);
  cudaMalloc((void**)&d_bitmap_opt, sizeof(int) * nPoints);

  cudaMemcpy(d_point, h_point, sizeof(float2) * nPoints, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vertex, h_vertex, sizeof(float2) * vertices, cudaMemcpyHostToDevice);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    pnpoly_base<<<(nPoints + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, BLOCK_SIZE_X>>>(d_bitmap_ref, d_point, d_vertex, nPoints);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float time;
  cudaEventElapsedTime(&time, start, end);
  printf("Average kernel execution time (pnpoly_base): %f (s)\n", time / repeat / 1000.0f);

  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    pnpoly_opt_1<<<(nPoints + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, BLOCK_SIZE_X>>>(d_bitmap_opt, d_point, d_vertex, nPoints);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);
  printf("Average kernel execution time (pnpoly_opt<1>): %f (s)\n", time / repeat / 1000.0f);

  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    pnpoly_opt_2<<<(nPoints + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, BLOCK_SIZE_X>>>(d_bitmap_opt, d_point, d_vertex, nPoints);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);
  printf("Average kernel execution time (pnpoly_opt<2>): %f (s)\n", time / repeat / 1000.0f);

  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    pnpoly_opt_4<<<(nPoints + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, BLOCK_SIZE_X>>>(d_bitmap_opt, d_point, d_vertex, nPoints);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);
  printf("Average kernel execution time (pnpoly_opt<4>): %f (s)\n", time / repeat / 1000.0f);

  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    pnpoly_opt_8<<<(nPoints + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, BLOCK_SIZE_X>>>(d_bitmap_opt, d_point, d_vertex, nPoints);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);
  printf("Average kernel execution time (pnpoly_opt<8>): %f (s)\n", time / repeat / 1000.0f);

  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    pnpoly_opt_16<<<(nPoints + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, BLOCK_SIZE_X>>>(d_bitmap_opt, d_point, d_vertex, nPoints);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);
  printf("Average kernel execution time (pnpoly_opt<16>): %f (s)\n", time / repeat / 1000.0f);

  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    pnpoly_opt_32<<<(nPoints + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, BLOCK_SIZE_X>>>(d_bitmap_opt, d_point, d_vertex, nPoints);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);
  printf("Average kernel execution time (pnpoly_opt<32>): %f (s)\n", time / repeat / 1000.0f);

  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    pnpoly_opt_64<<<(nPoints + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, BLOCK_SIZE_X>>>(d_bitmap_opt, d_point, d_vertex, nPoints);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);
  printf("Average kernel execution time (pnpoly_opt<64>): %f (s)\n", time / repeat / 1000.0f);

  cudaFree(d_point);
  cudaFree(d_vertex);
  cudaFree(d_bitmap_ref);
  cudaFree(d_bitmap_opt);

  free(h_point);
  free(h_vertex);

  return 0;
}