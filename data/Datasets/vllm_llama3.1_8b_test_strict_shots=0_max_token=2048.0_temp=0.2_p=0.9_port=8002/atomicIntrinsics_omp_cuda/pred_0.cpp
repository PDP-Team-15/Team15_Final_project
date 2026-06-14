```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda_runtime.h>

#include "reference.h"

template <class T>
__global__ void kernel(const int len, T* gpuData) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    atomicAdd(&gpuData[0], (T)10);
    atomicSub(&gpuData[1], (T)10);
    gpuData[4] &= (T)(2 * idx + 7);
    gpuData[5] |= (T)(1 << idx);
    gpuData[6] ^= (T)idx;
  }
}

template <class T>
void testcase(const int repeat)
{
  const int len = 1 << 10;
  unsigned int numThreads = 256;
  unsigned int numData = 7;
  unsigned int memSize = sizeof(T) * numData;
  const T data[] = {0, 0, (T)-256, 256, 255, 0, 255};
  T* gpuData;
  cudaMalloc((void**)&gpuData, memSize);
  cudaMemcpy(gpuData, data, memSize, cudaMemcpyHostToDevice);

  for (int n = 0; n < repeat; n++) {
    kernel<T><<<1, numThreads>>>(len, gpuData);
    cudaDeviceSynchronize();

    T maxVal = -numeric_limits<T>::max();
    T minVal = numeric_limits<T>::max();
    #pragma omp teams distribute parallel for thread_limit(256)
    for (int i = 0; i < len; ++i) {
      maxVal = max(maxVal, (T)i);
      minVal = min(minVal, (T)i);
    }

    cudaMemcpy(gpuData + 2, &maxVal, sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuData + 3, &minVal, sizeof(T), cudaMemcpyHostToDevice);
    kernel<T><<<1, numThreads>>>(len, gpuData);
    cudaDeviceSynchronize();

    cudaMemcpy(gpuData, data, memSize, cudaMemcpyDeviceToHost);
    computeGold<T>(gpuData, len);

    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      kernel<T><<<1, numThreads>>>(len, gpuData);
      cudaDeviceSynchronize();

      T maxVal = -numeric_limits<T>::max();
      T minVal = numeric_limits<T>::max();
      #pragma omp teams distribute parallel for thread_limit(256)
      for (int i = 0; i < len; ++i) {
        maxVal = max(maxVal, (T)i);
        minVal = min(minVal, (T)i);
      }

      cudaMemcpy(gpuData + 2, &maxVal, sizeof(T), cudaMemcpyHostToDevice);
      cudaMemcpy(gpuData + 3, &minVal, sizeof(T), cudaMemcpyHostToDevice);
      kernel<T><<<1, numThreads>>>(len, gpuData);
      cudaDeviceSynchronize();
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);
  }
  cudaFree(gpuData);
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  cudaSetDevice(0);
  cudaDeviceSynchronize();
  cudaStreamCreate(nullptr);
  testcase<int>(repeat);
  testcase<unsigned int>(repeat);
  cudaDeviceSynchronize();
  cudaStreamDestroy(nullptr);
  return 0;
}
```