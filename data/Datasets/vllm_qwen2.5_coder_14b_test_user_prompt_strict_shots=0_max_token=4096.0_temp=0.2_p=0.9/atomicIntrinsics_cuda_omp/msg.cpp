You are an HPC expert specializing in translating between parallel programming APIs. Output only the translated code inside a ```cpp ... ``` code block. No explanations, no comments outside the code block, no omissions.
For each kernel code provided, translate it from cuda to omp. Output ONLY the complete translated code wrapped in ```cpp ... ```. Do not truncate. Do not modify main(). All function names must match.
Translate the following code from cuda to omp. Output ONLY the complete translated code wrapped in ```cpp ... ```. Do not truncate. Do not modify main(). All function names must match. All functions, macros, and variables used in the output must be fully defined. Definitions must appear before their first use.
Translate the following code  from cuda to omp:
main-um.cu:






#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <cuda.h>
#include "kernel.h"
#include "reference.h"

template <class T>
void testcase(const int repeat)
{
  unsigned int len = 1 << 27;
  unsigned int numThreads = 256;
  unsigned int numBlocks = (len + numThreads - 1) / numThreads;
  T gpuData[] = {0, 0, (T)-256, 256, 255, 0, 255, 0, 0};
  unsigned int memSize = sizeof(gpuData);

  

  T *dOData;
  cudaMallocManaged((void **) &dOData, memSize);

  for (int i = 0; i < repeat; i++) {
    

    cudaMemcpy(dOData, gpuData, memSize, cudaMemcpyHostToDevice);

    

    testKernel<T><<<numBlocks, numThreads>>>(dOData);
  }
  cudaDeviceSynchronize();

  computeGold<T>(dOData, numThreads * numBlocks);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    

    testKernel<T><<<numBlocks, numThreads>>>(dOData);
  }
  cudaDeviceSynchronize();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  cudaFree(dOData);
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

main.cu:






#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <cuda.h>
#include "kernel.h"
#include "reference.h"

template <class T>
void testcase(const int repeat)
{
  unsigned int len = 1 << 27;
  unsigned int numThreads = 256;
  unsigned int numBlocks = (len + numThreads - 1) / numThreads;
  T gpuData[] = {0, 0, (T)-256, 256, 255, 0, 255, 0, 0};
  unsigned int memSize = sizeof(gpuData);

  

  T *dOData;
  cudaMalloc((void **) &dOData, memSize);

  for (int i = 0; i < repeat; i++) {
    

    cudaMemcpy(dOData, gpuData, memSize, cudaMemcpyHostToDevice);

    

    testKernel<T><<<numBlocks, numThreads>>>(dOData);
  }

  

  cudaMemcpy(gpuData, dOData, memSize, cudaMemcpyDeviceToHost);

  computeGold<T>(gpuData, numThreads * numBlocks);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    

    testKernel<T><<<numBlocks, numThreads>>>(dOData);
  }
  cudaDeviceSynchronize();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  cudaFree(dOData);
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
