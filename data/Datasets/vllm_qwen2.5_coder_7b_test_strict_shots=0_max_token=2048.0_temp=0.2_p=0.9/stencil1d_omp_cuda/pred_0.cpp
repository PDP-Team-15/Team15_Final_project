```cpp
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>

#define RADIUS 7
#define BLOCK_SIZE 256

__global__ void stencil_1d_kernel(int* a, int* b, int length) {
  __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < length + RADIUS) {
    temp[tid + RADIUS] = a[i];
    if (tid < RADIUS) {
      temp[tid] = (i < RADIUS) ? 0 : a[i - RADIUS];
      temp[tid + RADIUS + BLOCK_SIZE] = a[i + BLOCK_SIZE];
    }
  }

  __syncthreads();

  if (i < length) {
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++)
      result += temp[tid + RADIUS + offset];
    b[i] = result;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <length> <repeat>\n", argv[0]);
    printf("length is a multiple of %d\n", BLOCK_SIZE);
    return 1;
  }
  const int length = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  int size = length;
  int pad_size = (length + RADIUS);

  int* d_a, * d_b;
  cudaMalloc((void**)&d_a, pad_size * sizeof(int));
  cudaMalloc((void**)&d_b, size * sizeof(int));

  for (int i = 0; i < pad_size; i++) d_a[i] = i;

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    cudaMemcpy(d_a, d_a, pad_size * sizeof(int), cudaMemcpyHostToDevice);
    stencil_1d_kernel<<<(length + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, length);
    cudaMemcpy(d_b, d_b, size * sizeof(int), cudaMemcpyDeviceToHost);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  bool ok = true;
  for (int i = 0; i < 2 * RADIUS; i++) {
    int s = 0;
    for (int j = i; j <= i + 2 * RADIUS; j++)
      s += j < RADIUS ? 0 : (d_a[j] - RADIUS);
    if (s != d_b[i]) {
      printf("Error at %d: %d (host) != %d (device)\n", i, s, d_b[i]);
      ok = false;
      break;
    }
  }

  for (int i = 2 * RADIUS; i < length; i++) {
    int s = 0;
    for (int j = i - RADIUS; j <= i + RADIUS; j++)
      s += d_a[j];
    if (s != d_b[i]) {
      printf("Error at %d: %d (host) != %d (device)\n", i, s, d_b[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  cudaFree(d_a);
  cudaFree(d_b);
  return 0;
}
```