#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>

#define RADIUS 7
#define BLOCK_SIZE 256

// Kernel function to perform the stencil operation
__global__ void stencil_kernel(int* a, int* b, int length, int radius) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    int result = 0;
    for (int offset = -radius; offset <= radius; offset++) {
      int temp_idx = idx + radius + offset;
      if (temp_idx >= 0 && temp_idx < length) {
        result += a[temp_idx];
      }
    }
    b[idx] = result;
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

  int* a = (int *)malloc(pad_size * sizeof(int));
  int* b = (int *)malloc(size * sizeof(int));

  for (int i = 0; i < length + RADIUS; i++) a[i] = i;

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    int* d_a, * d_b;
    cudaMalloc((void**)&d_a, (pad_size + 1) * sizeof(int));
    cudaMalloc((void**)&d_b, (size + 1) * sizeof(int));
    cudaMemcpy(d_a, a, (pad_size + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, (size + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int num_blocks = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;
    stencil_kernel<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, length, RADIUS);
    cudaDeviceSynchronize();

    cudaMemcpy(b, d_b, (size + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
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

  free(a);
  free(b);
  return 0;
}