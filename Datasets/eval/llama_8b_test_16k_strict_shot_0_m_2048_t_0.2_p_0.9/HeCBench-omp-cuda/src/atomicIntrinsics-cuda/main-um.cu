#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <chrono>

#include "reference.h"

template <class T>
__global__ void kernel(const int len, T *gpuData) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    gpuData[0] += (T)10;
    gpuData[1] -= (T)10;
    gpuData[4] &= (T)(2 * idx + 7);
    gpuData[5] |= (T)(1 << idx);
    gpuData[6] ^= (T)idx;
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    gpuData[2] = max(gpuData[2], (T)idx);
    gpuData[3] = min(gpuData[3], (T)idx);
  }
}

template <class T>
void testcase(const int repeat) {
  const int len = 1 << 10;
  const int numThreads = 256;
  const int numData = 7;
  const T data[] = {0, 0, (T)-256, 256, 255, 0, 255};
  T gpuData[7];

  cudaMalloc((void **)&gpuData, sizeof(T) * numData);
  cudaMemcpy(gpuData, data, sizeof(T) * numData, cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int n = 0; n < repeat; n++) {
    cudaEventRecord(start, 0);

    kernel<<<(len + numThreads - 1) / numThreads, numThreads>>>(len, gpuData);

    cudaDeviceSynchronize();

    kernel<<<1, numThreads>>>(len, gpuData);

    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Average kernel execution time: %f (ms)\n", milliseconds / repeat);

    computeGold<T>(gpuData, len);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(gpuData);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  testcase<int>(repeat);
  testcase<unsigned int>(repeat);
  return 0;
}