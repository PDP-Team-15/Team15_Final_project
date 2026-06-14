```c
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
    // Copy data to device
    for (int j = 0; j < numBlocks; j++) {
      #pragma omp parallel for num_threads(numThreads)
      for (int k = 0; k < numThreads; k++) {
        int idx = j * numThreads + k;
        if (idx < len) {
          dOData[idx] = gpuData[idx];
        }
      }
    }

    // Launch kernel
    #pragma omp parallel for num_threads(numThreads)
    for (int j = 0; j < numBlocks; j++) {
      for (int k = 0; k < numThreads; k++) {
        int idx = j * numThreads + k;
        if (idx < len) {
          testKernel<T>(dOData + idx);
        }
      }
    }

    // Synchronize threads
    #pragma omp barrier
  }

  computeGold<T>(gpuData, numThreads * numBlocks);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    // Launch kernel
    #pragma omp parallel for num_threads(numThreads)
    for (int j = 0; j < numBlocks; j++) {
      for (int k = 0; k < numThreads; k++) {
        int idx = j * numThreads + k;
        if (idx < len) {
          testKernel<T>(dOData + idx);
        }
      }
    }

    // Synchronize threads
    #pragma omp barrier
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
  testcase<int>(repeat);
  testcase<unsigned int>(repeat);
  return 0;
}
```

Note: The `testKernel` function is assumed to be defined in the `kernel.h` header file. Also, the `computeGold` function is assumed to be defined in the `reference.h` header file. The `testKernel` function should be modified to take a pointer to an array of type `T` as an argument, and the `computeGold` function should be modified to take a pointer to an array of type `T` and the number of elements as arguments.