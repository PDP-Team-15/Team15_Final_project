Here is the equivalent CUDA code for the given OpenMP code:

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
```

Note that the kernel launch in CUDA is different from OpenMP. In CUDA, you need to specify the number of blocks and threads per block using the `<<<>>>` syntax. In this case, we are launching a single block with a single thread per block.

Also, note that we are using `cudaEvent` to measure the execution time of the kernels. This is because `cudaEvent` provides a more accurate way to measure the execution time compared to using `cudaDeviceSynchronize` and `cudaEventElapsedTime`.

The kernel function `pnpoly_base` and `pnpoly_opt` should be defined in the `kernel.h` file, and their implementation should be provided in the `kernel.cu` file.

Here is an example of how the `pnpoly_base` kernel function could be implemented:

```c
__global__ void pnpoly_base(int *bitmap, float2 *point, float2 *vertex, int nPoints) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nPoints) {
    int inside = 0;
    float x = point[idx].x;
    float y = point[idx].y;
    int i, j;
    for (i = 0, j = vertices - 1; i < vertices; j = i++) {
      if ((vertex[i].y > y) != (vertex[j].y > y) &&
          (x < (vertex[j].x - vertex[i].x) * (y - vertex[i].y) / (vertex[j].y - vertex[i].y) + vertex[i].x)) {
        inside = !inside;
      }
    }
    bitmap[idx] = inside;
  }
}
```

And here is an example of how the `pnpoly_opt` kernel function could be implemented:

```c
template <int BLOCK_SIZE>
__global__ void pnpoly_opt(int *bitmap, float2 *point, float2 *vertex, int nPoints) {
  __shared__ float2 vertex_shared[BLOCK_SIZE];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x < BLOCK_SIZE) {
    vertex_shared[threadIdx.x] = vertex[threadIdx.x];
  }
  __syncthreads();
  if (idx >= nPoints) return;
  int inside = 0;
  float x = point[idx].x;
  float y = point[idx].y;
  int i, j;
  for (i = 0, j = BLOCK_SIZE - 1; i < BLOCK_SIZE; j = i++) {
    if ((vertex_shared[i].y > y) != (vertex_shared[j].y > y) &&
        (x < (vertex_shared[j].x - vertex_shared[i].x) * (y - vertex_shared[i].y) / (vertex_shared[j].y - vertex_shared[i].y) + vertex_shared[i].x)) {
      inside = !inside;
    }
  }
  for (i = BLOCK_SIZE; i <