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

  computeGold<T>(gpuData, len);

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    atomicKernel1<<<1, numThreads, 0, 0>>>(d_gpuData, len);
    reductionMax<<<1, numThreads, 0, 0>>>(d_gpuData, len);
    reductionMin<<<1, numThreads, 0, 0>>>(d_gpuData, len);
  }

  cudaMemcpy(gpuData, d_gpuData, memSize, cudaMemcpyDeviceToHost);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  cudaFree(d_gpuData);
}

__global__ void atomicKernel1(T* gpuData, int len) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < len) {
    atomicAdd((T*)&gpuData[0], (T)10);
    atomicSub((T*)&gpuData[1], (T)10);
    atomicAnd((T*)&gpuData[4], (T)(2 * tid + 7));
    atomicOr((T*)&gpuData[5], (T)(1 << tid));
    atomicXor((T*)&gpuData[6], (T)tid);
  }
}

__global__ void reductionMax(T* gpuData, int len) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < len) {
    if (gpuData[2] < (T)tid) {
      atomicMax((T*)&gpuData[2], (T)tid);
    }
  }
}

__global__ void reductionMin(T* gpuData, int len) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < len) {
    if (gpuData[3] > (T)tid) {
      atomicMin((T*)&gpuData[3], (T)tid);
    }
  }
}

int main(int argc, char **argv)
{
  if (argc != 3) {
    printf("Usage: %s <number of atomic operations> <repeat>\n", argv[0]);
    return 1;
  }

  const int num = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  testcase<int>(num, repeat);
  testcase<unsigned int>(num, repeat);
  return 0;
}