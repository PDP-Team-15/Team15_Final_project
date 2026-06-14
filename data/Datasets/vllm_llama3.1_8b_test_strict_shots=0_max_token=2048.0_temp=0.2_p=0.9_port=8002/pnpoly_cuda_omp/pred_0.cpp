```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <random>
#include <chrono>

#define VERTICES 600
#define BLOCK_SIZE_X 256

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

  float2 *d_point;
  float2 *d_vertex;
  int *d_bitmap_ref, *d_bitmap_opt;

  d_point = (float2*) malloc (nPoints*sizeof(float2));
  d_vertex = (float2*) malloc (vertices*sizeof(float2));
  d_bitmap_ref = (int*) malloc (nPoints*sizeof(int));
  d_bitmap_opt = (int*) malloc (nPoints*sizeof(int));

  int num_threads = omp_get_max_threads();
  int num_blocks = (nPoints+num_threads-1)/num_threads;

  for (int i = 0; i < nPoints; i++) {
    d_point[i].x = point[i].x;
    d_point[i].y = point[i].y;
  }

  for (int i = 0; i < vertices; i++) {
    d_vertex[i].x = vertex[i].x;
    d_vertex[i].y = vertex[i].y;
  }

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    #pragma omp parallel for num_threads(num_threads)
    for (int j = 0; j < num_blocks; j++) {
      int block_start = j * num_threads;
      int block_end = (j+1) * num_threads;
      if (block_end > nPoints) block_end = nPoints;
      pnpoly_base(d_bitmap_ref + block_start, d_point + block_start, d_vertex, nPoints, block_end);
    }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_base): %f (s)\n", (time * 1e-9f) / repeat);

  for (int i = 0; i < nPoints; i++) bitmap_ref[i] = d_bitmap_ref[i];

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    #pragma omp parallel for num_threads(num_threads)
    for (int j = 0; j < num_blocks; j++) {
      int block_start = j * num_threads;
      int block_end = (j+1) * num_threads;
      if (block_end > nPoints) block_end = nPoints;
      pnpoly_opt<1>(d_bitmap_opt + block_start, d_point + block_start, d_vertex, nPoints, block_end);
    }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<1>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    #pragma omp parallel for num_threads(num_threads)
    for (int j = 0; j < num_blocks; j++) {
      int block_start = j * num_threads;
      int block_end = (j+1) * num_threads;
      if (block_end > nPoints) block_end = nPoints;
      pnpoly_opt<2>(d_bitmap_opt + block_start, d_point + block_start, d_vertex, nPoints, block_end);
    }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<2>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    #pragma omp parallel for num_threads(num_threads)
    for (int j = 0; j < num_blocks; j++) {
      int block_start = j * num_threads;
      int block_end = (j+1) * num_threads;
      if (block_end > nPoints) block_end = nPoints;
      pnpoly_opt<4>(d_bitmap_opt + block_start, d_point + block_start, d_vertex, nPoints, block_end);
    }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<4>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    #pragma omp parallel for num_threads(num_threads)
    for (int j = 0; j < num_blocks; j++) {
      int block_start = j * num_threads;
      int block_end = (j+1) * num_threads;
      if (block_end > nPoints) block_end = nPoints;
      pnpoly_opt<8>(d_bitmap_opt + block_start, d_point + block_start, d_vertex, nPoints, block_end);
    }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<8>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    #pragma omp parallel for num_threads(num_threads)
    for (int j = 0; j < num_blocks; j++) {
      int block_start = j * num_threads;
      int block_end = (j+1) * num_threads;
      if (block_end > nPoints) block_end = nPoints;
      pnpoly_opt<16>(d_bitmap_opt + block_start, d_point + block_start, d_vertex, nPoints, block_end);
    }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<16>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    #pragma omp parallel for num_threads(num_threads)
    for (int j = 0; j < num_blocks; j++) {
      int block_start = j * num_threads;
      int block_end = (j+1) * num_threads;
      if (block_end > nPoints) block_end = nPoints;
      pnpoly_opt<32>(d_bitmap_opt + block_start, d_point + block_start, d_vertex, nPoints, block_end);
    }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<32>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    #pragma omp parallel for num_threads(num_threads)
    for (int j = 0; j < num_blocks; j++) {
      int block_start = j * num_threads;
      int block_end = (j+1) * num_threads;
      if (block_end > nPoints) block_end = nPoints;
      pnpoly_opt<64>(d_bitmap_opt + block_start, d_point + block_start, d_vertex, nPoints, block_end);
    }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration