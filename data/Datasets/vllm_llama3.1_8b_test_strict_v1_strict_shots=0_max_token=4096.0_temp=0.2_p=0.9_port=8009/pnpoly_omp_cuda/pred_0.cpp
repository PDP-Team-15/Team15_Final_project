```cuda
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

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    pnpoly_base<<<1, 256>>>(d_bitmap_ref, d_point, d_vertex, nPoints);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_base): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    pnpoly_opt<1><<<1, 256>>>(d_bitmap_opt, d_point, d_vertex, nPoints);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<1>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    pnpoly_opt<2><<<1, 256>>>(d_bitmap_opt, d_point, d_vertex, nPoints);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<2>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    pnpoly_opt<4><<<1, 256>>>(d_bitmap_opt, d_point, d_vertex, nPoints);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<4>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    pnpoly_opt<8><<<1, 256>>>(d_bitmap_opt, d_point, d_vertex, nPoints);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<8>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    pnpoly_opt<16><<<1, 256>>>(d_bitmap_opt, d_point, d_vertex, nPoints);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<16>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    pnpoly_opt<32><<<1, 256>>>(d_bitmap_opt, d_point, d_vertex, nPoints);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<32>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    pnpoly_opt<64><<<1, 256>>>(d_bitmap_opt, d_point, d_vertex, nPoints);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<64>): %f (s)\n", (time * 1e-9f) / repeat);

  cudaMemcpy(h_bitmap_opt, d_bitmap_opt, nPoints * sizeof(int), cudaMemcpyDeviceToHost);

  int error = memcmp(h_bitmap_opt, h_bitmap_ref, nPoints*sizeof(int)); 
  
  

  int checksum = 0;
  for (int i = 0; i < nPoints; i++) checksum += h_bitmap_opt[i];
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
```

```cuda
// kernel.h
#ifndef KERNEL_H
#define KERNEL_H

__global__ void pnpoly_base(int *bitmap, float2 *point, float2 *vertex, int nPoints);

template<int N>
__global__ void pnpoly_opt(int *bitmap, float2 *point, float2 *vertex, int nPoints);

#endif
```

```cuda
// kernel.cu
#include "kernel.h"

__global__ void pnpoly_base(int *bitmap, float2 *point, float2 *vertex, int nPoints) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nPoints) {
    int inside = 0;
    float x = point[idx].x;
    float y = point[idx].y;
    float2 v0 = vertex[0];
    float2 v1 = vertex[1];
    float2 v2 = vertex[2];
    float2 v3 = vertex[3];
    float2 v4 = vertex[4];
    float2 v5 = vertex[5];
    float2 v6 = vertex[6];
    float2 v7 = vertex[7];
    float2 v8 = vertex[8];
    float2 v9 = vertex[9];
    float2 v10 = vertex[10];
    float2 v11 = vertex[11];
    float2 v12 = vertex[12];
    float2 v13 = vertex[13];
    float2 v14 = vertex[14];
    float2 v15 = vertex[15];
    float2 v16 = vertex[16];
    float2 v17 = vertex[17];
    float2 v18 = vertex[18];
    float2 v19 = vertex[19];
    float2 v20 = vertex[20];
    float2 v21 = vertex[21];
    float2 v22 = vertex[22];
    float2 v23 = vertex[23];
    float2 v24 = vertex[24];
    float2 v25 = vertex[25];
    float2 v26 = vertex[26];
    float2 v27 = vertex[27];
    float2 v28 = vertex[28];
    float2 v29 = vertex[29];
    float2 v30 = vertex[30];
    float2 v31 = vertex[31];
    float2 v32 = vertex[32];
    float2 v33 = vertex[33];
    float2 v34 = vertex[34];
    float2 v35 = vertex[35];
    float2 v36 = vertex[36];
    float2 v37 = vertex[37];
    float2 v38 = vertex[38];
    float2 v39 = vertex[39];
    float2 v40 = vertex[40];
    float2 v41 = vertex[41];
    float2 v42 = vertex[42];
    float2 v43 = vertex[43];
    float2 v44 = vertex[44];
    float2 v45 = vertex[45];
    float2 v46 = vertex[46];
    float2 v47 = vertex[47];
    float2 v48 = vertex[48];
    float2 v49 = vertex[49];
    float2 v50 = vertex[50];
    float2 v51 = vertex[51];
    float2 v52 = vertex[52];
    float2 v53 = vertex[53];
    float2 v54 = vertex[54];
    float2 v55 = vertex[55];
    float2 v56 = vertex[56];
    float2 v57 = vertex[57];
    float2 v58 = vertex[58];
    float2 v59 = vertex[59];
    float2 v60 = vertex[60];
    float2 v61 = vertex[61];
    float2 v62 = vertex[62];
    float2 v63 = vertex[63];

    for (int i = 0; i < 64; i++) {
      float2 v = vertex[i];
      if (v.x == 0 && v.y == 0) continue;
      float2 p = {x, y};
      float2 q = {v.x, v.y};
      float2 r = {p.x - q.x, p.y - q.y};
      float2 s = {r.x * r.x + r.y * r.y, 0};
      if (s.x < 0) continue;
      float2 t = {0, 0};
      if (s.x > 0) {
        t = {r.x * (r.x - 2 * q.x) + q.x * q.x, r.y * (r.y - 2 * q.y) + q.y * q.y};
        if (t.x < 0) continue;
        s = t;
      }
      if (s.x > 0) {
        t = {r.x * (r.x - 2 * q.x) + q.x * q.x, r.y * (r.y - 2 * q.y) + q.y * q.y};
        if (t.x < 0) continue;
        s = t;
      }
      if (s.x > 0) {
        t = {r.x * (r.x - 2 * q.x) + q.x * q.x, r.y * (r.y - 2 * q.y) + q.y * q.y};
        if (t.x < 0) continue;
        s = t;
      }
      if (s.x > 0) {
        t = {r.x * (r.x - 2 * q.x) + q.x * q.x, r.y * (r.y - 2 * q.y) + q.y * q.y};
        if (t.x < 0) continue;
        s = t;
      }
      if (s.x > 0) {
        t = {r.x * (r.x - 2 * q.x) + q.x * q.x, r.y * (r.y - 2 * q.y) + q.y * q.y};
        if (t.x < 0) continue;
        s = t;
      }
      if (s.x > 0) {
        t = {r.x * (r.x - 2 * q.x) + q.x * q.x, r.y * (r.y - 2 * q.y) + q.y * q.y};
        if (t.x < 0) continue;
        s = t;
      }
      if (s.x > 0) {
        t = {r.x * (r.x - 2 * q.x) + q.x * q.x, r.y * (r.y - 2 * q.y) + q.y * q.y};
        if (t.x < 0) continue;
        s = t;
      }
      if (s.x > 0) {
        t = {r.x * (r.x - 2 * q.x) + q.x * q.x, r.y * (r.y - 2 * q.y) + q.y * q.y};
        if (t.x < 0) continue;
        s = t;
      }
      if (s.x > 0) {
        t = {r.x * (r.x - 2 * q.x) + q.x * q.x, r.y * (r.y - 2 * q.y) + q.y * q.y};
        if (t.x < 0) continue;
        s = t;
      }
      if (s.x > 0) {
        t = {r.x * (r.x - 2 * q.x) + q.x * q.x, r.y * (r.y - 2 * q.y) + q.y * q.y};
        if (t.x < 0) continue;
        s = t;
      }
      if (s.x > 0) {
        t = {r.x * (r.x - 2 * q.x) + q.x * q.x, r.y * (r.y - 2 * q.y) + q.y * q.y};
        if (t.x < 0) continue;
        s = t;
      }
      if (s.x > 0) {
        t = {r.x * (r.x - 2 * q.x) + q.x * q.x, r.y * (r.y - 2 * q.y) + q.y * q.y};
        if (t.x < 0) continue;
        s = t;
      }
      if (s.x > 0) {
        t = {r.x * (r.x - 2 * q.x) + q.x * q.x, r.y * (r.y - 2 * q.y) + q.y * q.y};
        if (t.x < 0) continue;
        s = t;
      }
      if (s.x > 0) {
        t = {r.x * (r.x - 2 * q.x) + q.x * q.x, r.y * (r.y - 2 * q.y) + q.y * q.y};
        if (t.x < 0) continue;
        s = t;
      }
      if (s.x > 0) {
        t = {r.x * (r.x - 2 * q.x) + q.x * q.x, r.y * (r.y - 2 * q.y) + q.y * q.y};
        if (t.x < 0) continue;
        s = t;
      }
      if (s.x > 0) {
        t = {r.x * (r.x - 2 * q.x) + q.x * q.x, r.y * (r.y - 2 * q.y) + q.y * q.y};
        if (t.x < 0) continue;
        s = t;
      }
      if (s.x > 0) {
        t = {r.x * (r.x - 2 * q.x) + q.x * q.x, r.y * (r.y - 2 * q.y) + q.y * q.y};
        if (t.x < 0) continue;
        s = t;
      }
      if (s.x > 0) {
        t = {r.x * (r.x - 2 * q.x) + q.x * q.x, r.y * (r.y - 2 * q.y) + q.y * q.y};
        if (t.x < 0) continue;
        s = t;
      }
      if (s.x > 0) {
        t = {r.x * (r.x - 2 * q.x) + q.x * q.x, r.y * (r.y - 2 * q.y) + q.y * q.y};
        if (t.x < 0) continue;
        s = t;
      }
      if (s.x > 0) {
        t = {r.x * (r.x - 2 * q.x) + q.x * q.x, r.y * (r.y - 2 * q.y) + q.y * q.y};
        if (t.x < 0) continue;
        s = t;
      }
      if (s.x > 0) {
        t = {r.x * (r.x - 2 * q.x) + q.x * q.x, r.y * (r.y - 2 * q.y) + q.y * q.y};
        if (t.x < 0) continue;
        s = t;
      }
     