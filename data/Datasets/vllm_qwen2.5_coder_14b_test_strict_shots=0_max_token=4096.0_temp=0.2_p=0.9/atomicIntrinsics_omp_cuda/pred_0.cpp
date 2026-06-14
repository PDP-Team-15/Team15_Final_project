```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda_runtime.h>

#include "reference.h"

template <class T>
__global__ void kernel(T* gpuData, int len)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len)
  {
    atomicAdd(&gpuData[0], (T)10);
    atomicSub(&gpuData[1], (T)10);
    atomicAnd(&gpuData[4], (T)(2 * i + 7));
    atomicOr(&gpuData[5], (T)(1 << i));
    atomicXor(&gpuData[6], (T)i);
  }
}

template <class T>
__global__ void maxKernel(T* gpuData, int len)
{
  extern __shared__ T sdata[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len)
    sdata[threadIdx.x] = gpuData[2];
  else
    sdata[threadIdx.x] = (T)-1;

  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (threadIdx.x < s)
      sdata[threadIdx.x] = max(sdata[threadIdx.x], sdata[threadIdx.x + s]);
    __syncthreads();
  }

  if (threadIdx.x == 0)
    gpuData[2] = sdata[0];
}

template <class T>
__global__ void minKernel(T* gpuData, int len)
{
  extern __shared__ T sdata[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len)
    sdata[threadIdx.x] = gpuData[3];
  else
    sdata[threadIdx.x] = (T)INT_MAX;

  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (threadIdx.x < s)
      sdata[threadIdx.x] = min(sdata[threadIdx.x], sdata[threadIdx.x + s]);
    __syncthreads();
  }

  if (threadIdx.x == 0)
    gpuData[3] = sdata[0];
}

template <class T>
void testcase(const int repeat)
{
  const int len = 1 << 10;
  unsigned int numThreads = 256;
  unsigned int numBlocks = (len + numThreads - 1) / numThreads;
  unsigned int numData = 7;
  unsigned int memSize = sizeof(T) * numData;
  const T data[] = {0, 0, (T)-256, 256, 255, 0, 255};
  T* gpuData;
  cudaMalloc(&gpuData, memSize);
  cudaMemcpy(gpuData, data, memSize, cudaMemcpyHostToDevice);

  for (int n = 0; n < repeat; n++)
  {
    kernel<<<numBlocks, numThreads>>>(gpuData, len);
    maxKernel<<<numBlocks, numThreads, numThreads * sizeof(T)>>>(gpuData, len);
    minKernel<<<numBlocks, numThreads, numThreads * sizeof(T)>>>(gpuData, len);
  }

  cudaMemcpy(gpuData, data, memSize, cudaMemcpyHostToDevice);

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++)
  {
    kernel<<<numBlocks, numThreads>>>(gpuData, len);
    maxKernel<<<numBlocks, numThreads, numThreads * sizeof(T)>>>(gpuData, len);
    minKernel<<<numBlocks, numThreads, numThreads * sizeof(T)>>>(gpuData, len);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  cudaMemcpy(gpuData, data, memSize, cudaMemcpyHostToDevice);
  computeGold<T>(gpuData, len);
  cudaFree(gpuData);
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  testcase<int>(repeat);
  testcase<unsigned int>(repeat);
  return 0;
}
```