```cpp
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <chrono>
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
  dOData = (T*)malloc(memSize);

  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams distribute parallel for map(to: dOData[0:memSize])
    for (int j = 0; j < memSize; j++) {
      dOData[j] = gpuData[j];
    }

    #pragma omp target teams distribute parallel for map(from: dOData[0:memSize])
    for (int j = 0; j < memSize; j++) {
      gpuData[j] = dOData[j];
    }

    #pragma omp target teams distribute parallel for map(to: dOData[0:memSize])
    for (int j = 0; j < memSize; j++) {
      gpuData[j] = testKernel<T>(dOData[j]);
    }
  }

  computeGold<T>(gpuData, numThreads * numBlocks);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams distribute parallel for map(to: dOData[0:memSize])
    for (int j = 0; j < memSize; j++) {
      gpuData[j] = testKernel<T>(dOData[j]);
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  free(dOData);
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  #pragma omp parallel for
  for (int i = 0; i < 2; i++) {
    testcase<int>(repeat);
    testcase<unsigned int>(repeat);
  }
  return 0;
}
```