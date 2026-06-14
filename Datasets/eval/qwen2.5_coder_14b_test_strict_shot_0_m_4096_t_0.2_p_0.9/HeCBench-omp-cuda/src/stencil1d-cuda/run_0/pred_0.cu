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

__global__ void stencilKernel(int* a, int* b, int length) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int temp[BLOCK_SIZE + 2 * RADIUS];

  if (idx < length) {
    temp[threadIdx.x + RADIUS] = a[idx];
    if (threadIdx.x < RADIUS) {
      temp[threadIdx.x] = (idx < RADIUS) ? 0 : a[idx - RADIUS];
      temp[threadIdx.x + RADIUS + BLOCK_SIZE] = a[idx + BLOCK_SIZE];
    }

    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++)
      result += temp[threadIdx.x + RADIUS + offset];
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

  int size = length * sizeof(int);
  int pad_size = (length + RADIUS) * sizeof(int);

  int *a, *b;
  // Alloc space for host copies of a, b, c and setup input values
  a = (int *)malloc(pad_size); 
  b = (int *)malloc(size);

  for (int i = 0; i < length+RADIUS; i++) a[i] = i;

  int *d_a, *d_b;
  // Alloc space for device copies of a, b, c
  cudaMalloc((void **)&d_a, pad_size);
  cudaMalloc((void **)&d_b, size);

  // Copy inputs to device
  cudaMemcpy(d_a, a, pad_size, cudaMemcpyHostToDevice);

  dim3 grids (length/BLOCK_SIZE);
  dim3 blocks (BLOCK_SIZE);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  // Launch kernel on GPU
  for (int i = 0; i < repeat; i++)
    stencil_1d <<< grids, blocks >>> (d_a, d_b);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  // Copy result back to host
  cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

  // verification
  bool ok = true;
  for (int i = 0; i < 2*RADIUS; i++) {
    int s = 0;
    for (int j = i; j <= i+2*RADIUS; j++)
      s += j < RADIUS ? 0 : (a[j] - RADIUS);
    if (s != b[i]) {
      printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
      ok = false;
      break;
    }
  }

  for (int i = 2*RADIUS; i < length; i++) {
    int s = 0;
    for (int j = i-RADIUS; j <= i+RADIUS; j++)
      s += a[j];
    if (s != b[i]) {
      printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  // Cleanup
  free(a);
  free(b); 
  cudaFree(d_a); 
  cudaFree(d_b); 
  return 0;
}

Error code: 400 - {'object': 'error', 'message': "This model's maximum context length is 8192 tokens. However, you requested 16163 tokens (1163 in the messages, 15000 in the completion). Please reduce the length of the messages or completion.", 'type': 'BadRequestError', 'param': None, 'code': 400}