```cpp
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

#define RADIUS 7
#define BLOCK_SIZE 256

__global__ void stencil_kernel(int* a, int* b, int length, int radius) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= length) return;

  int temp[BLOCK_SIZE + 2 * RADIUS];
  for (int j = 0; j < BLOCK_SIZE; j++) {
    int gindex = idx + j;
    temp[j + RADIUS] = a[gindex];
    if (j < RADIUS) {
      temp[j] = (gindex < RADIUS) ? 0 : a[gindex - RADIUS];
      temp[j + RADIUS + BLOCK_SIZE] = a[gindex + BLOCK_SIZE];
    }
  }

  for (int j = 0; j < BLOCK_SIZE; j++) {
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++)
      result += temp[j + RADIUS + offset];
    b[idx + j] = result;
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

  int* a;
  int* b;
  cudaMallocManaged(&a, (pad_size + size) * sizeof(int));
  cudaMallocManaged(&b, size * sizeof(int));

  for (int i = 0; i < length + RADIUS; i++) a[i] = i;

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    stencil_kernel<<<(length + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, b, length, RADIUS);
    cudaDeviceSynchronize();
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  bool ok = true;
  for (int i = 0; i < 2 * RADIUS; i++) {
    int s = 0;
    for (int j = i; j <= i + 2 * RADIUS; j++)
      s += j < RADIUS ? 0 : (a[j] - RADIUS);
    if (s != b[i]) {
      printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
      ok = false;
      break;
    }
  }

  for (int i = 2 * RADIUS; i < length; i++) {
    int s = 0;
    for (int j = i - RADIUS; j <= i + RADIUS; j++)
      s += a[j];
    if (s != b[i]) {
      printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  cudaFree(a);
  cudaFree(b);
  return 0;
}
```