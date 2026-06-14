#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

#include <cuda_runtime.h>
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

  T* d_gpuData;
  cudaMalloc((void**)&d_gpuData, memSize);
  cudaMemcpy(d_gpuData, data, memSize, cudaMemcpyHostToDevice);

  for (int n = 0; n < repeat; n++) {
    cudaMemcpy(d_gpuData, data, memSize, cudaMemcpyHostToDevice);

    dim3 threads(numThreads);
    dim3 blocks((len + numThreads - 1) / numThreads);

    kernel1<<<blocks, threads>>>(d_gpuData, len);
    kernel2<<<blocks, threads>>>(d_gpuData, len);
    kernel3<<<blocks, threads>>>(d_gpuData, len);
  }

  cudaMemcpy(gpuData, d_gpuData, memSize, cudaMemcpyDeviceToHost);
  computeGold<T>(gpuData, len);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  for (int n = 0; n < repeat; n++) {
    dim3 threads(numThreads);
    dim3 blocks((len + numThreads - 1) / numThreads);

    kernel1<<<blocks, threads>>>(d_gpuData, len);
    kernel2<<<blocks, threads>>>(d_gpuData, len);
    kernel3<<<blocks, threads>>>(d_gpuData, len);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float time;
  cudaEventElapsedTime(&time, start, stop);
  printf("Average kernel execution time: %f (us)\n", time / repeat);

  cudaFree(d_gpuData);
}

__global__ void kernel1(T* gpuData, int len)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    atomicAdd((T*)&gpuData[0], (T)10);
    atomicSub((T*)&gpuData[1], (T)10);
    atomicAnd((T*)&gpuData[4], (T)(2*i+7));
    atomicOr((T*)&gpuData[5], (T)(1 << i));
    atomicXor((T*)&gpuData[6], (T)i);
  }
}

__global__ void kernel2(T* gpuData, int len)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    if (i > gpuData[2]) {
      gpuData[2] = i;
    }
  }
}

__global__ void kernel3(T* gpuData, int len)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    if (i < gpuData[3]) {
      gpuData[3] = i;
    }
  }
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