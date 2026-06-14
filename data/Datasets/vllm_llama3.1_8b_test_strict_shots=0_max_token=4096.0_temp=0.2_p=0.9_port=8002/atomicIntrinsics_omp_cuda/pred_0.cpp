```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda_runtime.h>

#include "reference.h"

template <class T>
__global__ void kernel(const T* data, T* gpuData, const int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    atomicAdd(&gpuData[0], (T)10);
    atomicSub(&gpuData[1], (T)10);
    gpuData[4] = gpuData[4] & (T)(2 * idx + 7);
    gpuData[5] = gpuData[5] | (T)(1 << idx);
    gpuData[6] = gpuData[6] ^ (T)idx;
  }
}

template <class T>
void testcase(const int repeat) {
  const int len = 1 << 10;
  const int numThreads = 256;
  const int numData = 7;
  const int memSize = sizeof(T) * numData;
  T* data = (T*)malloc(memSize);
  T* gpuData;
  cudaMalloc((void**)&gpuData, memSize);

  T* d_data;
  cudaMalloc((void**)&d_data, memSize);

  T goldData[7];

  for (int n = 0; n < repeat; n++) {
    cudaMemcpy(d_data, data, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuData, data, memSize, cudaMemcpyHostToDevice);

    kernel<T><<<1, numThreads>>>(d_data, gpuData, len);
    cudaDeviceSynchronize();

    kernel<T><<<1, numThreads>>>(d_data, gpuData, len);
    cudaDeviceSynchronize();

    cudaMemcpy(gpuData, data, memSize, cudaMemcpyDeviceToHost);

    computeGold<T>(gpuData, len);

    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      kernel<T><<<1, numThreads>>>(d_data, gpuData, len);
      cudaDeviceSynchronize();

      kernel<T><<<1, numThreads>>>(d_data, gpuData, len);
      cudaDeviceSynchronize();
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);
  }

  free(data);
  cudaFree(gpuData);
  cudaFree(d_data);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  cudaSetDevice(0);
  cudaDeviceSynchronize();
  testcase<int>(repeat);
  testcase<unsigned int>(repeat);
  return 0;
}
```