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

  float2 *d_point;
  float2 *h_point = (float2*) malloc (sizeof(float2) * nPoints);
  for (int i = 0; i < nPoints; i++) {
    h_point[i].x = distribution(rng);
    h_point[i].y = distribution(rng);
  }

  float2 *d_vertex;
  float2 *h_vertex = (float2*) malloc (vertices * sizeof(float2));
  for (int i = 0; i < vertices; i++) {
    float t = distribution(rng) * 2.f * M_PI;
    h_vertex[i].x = cosf(t);
    h_vertex[i].y = sinf(t);
  }

  int *d_bitmap_ref;
  int *d_bitmap_opt;
  int *h_bitmap_ref = (int*) malloc (nPoints * sizeof(int));
  int *h_bitmap_opt = (int*) malloc (nPoints * sizeof(int));

  cudaMalloc((void**)&d_point, nPoints * sizeof(float2));
  cudaMalloc((void**)&d_vertex, vertices * sizeof(float2));
  cudaMalloc((void**)&d_bitmap_ref, nPoints * sizeof(int));
  cudaMalloc((void**)&d_bitmap_opt, nPoints * sizeof(int));

  cudaMemcpy(d_point, h_point, nPoints * sizeof(float2), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vertex, h_vertex, vertices * sizeof(float2), cudaMemcpyHostToDevice);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  for (int i = 0; i < repeat; i++) {
    cudaEventRecord(start, 0);
    pnpoly_base<<<256, 256>>>(d_bitmap_ref, d_point, d_vertex, nPoints);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    printf("Average kernel execution time (pnpoly_base): %f (ms)\n", time / repeat);
  }

  for (int i = 0; i < repeat; i++) {
    cudaEventRecord(start, 0);
    pnpoly_opt<1><<<256, 256>>>(d_bitmap_opt, d_point, d_vertex, nPoints);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    printf("Average kernel execution time (pnpoly_opt<1>): %f (ms)\n", time / repeat);
  }

  for (int i = 0; i < repeat; i++) {
    cudaEventRecord(start, 0);
    pnpoly_opt<2><<<256, 256>>>(d_bitmap_opt, d_point, d_vertex, nPoints);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    printf("Average kernel execution time (pnpoly_opt<2>): %f (ms)\n", time / repeat);
  }

  for (int i = 0; i < repeat; i++) {
    cudaEventRecord(start, 0);
    pnpoly_opt<4><<<256, 256>>>(d_bitmap_opt, d_point, d_vertex, nPoints);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    printf("Average kernel execution time (pnpoly_opt<4>): %f (ms)\n", time / repeat);
  }

  for (int i = 0; i < repeat; i++) {
    cudaEventRecord(start, 0);
    pnpoly_opt<8><<<256, 256>>>(d_bitmap_opt, d_point, d_vertex, nPoints);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    printf("Average kernel execution time (pnpoly_opt<8>): %f (ms)\n", time / repeat);
  }

  for (int i = 0; i < repeat; i++) {
    cudaEventRecord(start, 0);
    pnpoly_opt<16><<<256, 256>>>(d_bitmap_opt, d_point, d_vertex, nPoints);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    printf("Average kernel execution time (pnpoly_opt<16>): %f (ms)\n", time / repeat);
  }

  for (int i = 0; i < repeat; i++) {
    cudaEventRecord(start, 0);
    pnpoly_opt<32><<<256, 256>>>(d_bitmap_opt, d_point, d_vertex, nPoints);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    printf("Average kernel execution time (pnpoly_opt<32>): %f (ms)\n", time / repeat);
  }

  for (int i = 0; i < repeat; i++) {
    cudaEventRecord(start, 0);
    pnpoly_opt<64><<<256, 256>>>(d_bitmap_opt, d_point, d_vertex, nPoints);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    printf("Average kernel execution time (pnpoly_opt<64>): %f (ms)\n", time / repeat);
  }

  int error = memcmp(h_bitmap_opt, h_bitmap_ref, nPoints*sizeof(int)); 
  
  int checksum = 0;
  for (int i = 0; i < nPoints; i++) checksum += h_bitmap_opt[i];
  printf("Checksum: %d\n", checksum);

  printf("%s\n", error ? "FAIL" : "PASS");

  free(h_point);
  free(h_vertex);
  free(h_bitmap_ref);
  free(h_bitmap_opt);
  cudaFree(d_point);
  cudaFree(d_vertex);
  cudaFree(d_bitmap_ref);
  cudaFree(d_bitmap_opt);
  return 0;
}