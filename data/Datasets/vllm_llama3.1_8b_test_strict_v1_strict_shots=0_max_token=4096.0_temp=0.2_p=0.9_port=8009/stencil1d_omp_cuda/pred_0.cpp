```cuda
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>

#define RADIUS 7
#define BLOCK_SIZE 256

__global__ void stencil_kernel(int* a, int* b, int length, int radius) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= length) return;

  int temp[BLOCK_SIZE + 2 * RADIUS];
  temp[idx + RADIUS] = a[idx];

  if (idx < radius) {
    temp[idx] = (idx < radius) ? 0 : a[idx - radius];
    temp[idx + RADIUS + BLOCK_SIZE] = a[idx + BLOCK_SIZE];
  }

  int result = 0;
  for (int offset = -radius; offset <= radius; offset++)
    result += temp[idx + RADIUS + offset];

  b[idx] = result;
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

Note that I used `cudaMallocManaged` to allocate memory on the host and device, and `cudaDeviceSynchronize` to ensure that the kernel has finished executing before measuring the execution time. I also removed the `#pragma omp target` directive, as it is not necessary in CUDA.