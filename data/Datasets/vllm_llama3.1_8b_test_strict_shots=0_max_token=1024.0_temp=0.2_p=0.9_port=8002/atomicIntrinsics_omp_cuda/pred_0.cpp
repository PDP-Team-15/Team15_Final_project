```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <chrono>

#include "reference.h"

template <class T>
void testcase(const int repeat)
{
  const int len = 1 << 10;
  unsigned int numThreads = 256;
  unsigned int numData = 7;
  unsigned int memSize = sizeof(T) * numData;
  const T data[] = {0, 0, (T)-256, 256, 255, 0, 255};
  T gpuData[7];

  cudaEvent_t start, stop;
  float time;

  cudaMalloc((void**)&gpuData, sizeof(T) * numData);
  cudaMemcpy(gpuData, data, memSize, cudaMemcpyHostToDevice);

  for (int n = 0; n < repeat; n++) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    kernel<<<(len + 255 - 1) / 256, 256>>>(gpuData, len);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Kernel execution time: %f (ms)\n", time);

    kernelReductionMax<<<(len + 255 - 1) / 256, 256>>>(gpuData, len);
    kernelReductionMin<<<(len + 255 - 1) / 256, 256>>>(gpuData, len);

    cudaDeviceSynchronize();

    computeGold<T>(gpuData, len);

    for (int i = 0; i < len; ++i)
      gpuData[0] += (T)10;
    gpuData[1] -= (T)10;
    gpuData[4] &= (T)(2 * i + 7);
    gpuData[5] |= (T)(1 << i);
    gpuData[6] ^= (T)i;

    kernel<<<(len + 255 - 1) / 256, 256>>>(gpuData, len);

    kernelReductionMax<<<(len + 255 - 1) / 256, 256>>>(gpuData, len);
    kernelReductionMin<<<(len + 255 - 1) / 256, 256>>>(gpuData, len);

    cudaDeviceSynchronize();
  }

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    kernel<<<(len + 255 - 1) / 256, 256>>>(gpuData, len);

    kernelReductionMax<<<(len + 255 - 1) / 256, 256>>>(gpuData, len);
    kernelReductionMin<<<(len + 255 - 1) / 256, 256>>>(gpuData, len);

    cudaDeviceSynchronize();
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  cudaFree(gpuData);
}

__global__ void kernel(T *gpuData, const int len)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    gpuData[0] += (T)10;
    gpuData[1] -= (T)10;
    gpuData[4] &= (T)(2 * idx + 7);
    gpuData[5] |= (T)(1 << idx);
    gpuData[6] ^= (T)idx;
  }
}

__global__ void kernelReductionMax(T *gpuData, const int len)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    gpuData[2] = max(gpuData[2], (T)idx);
  }
}

__global__ void kernelReductionMin(T *gpuData, const int len)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    gpuData[3] = min(gpuData[3], (T)idx);
  }
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  cudaSetDevice(0);
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  cudaDeviceSetTccDriver(1);
 