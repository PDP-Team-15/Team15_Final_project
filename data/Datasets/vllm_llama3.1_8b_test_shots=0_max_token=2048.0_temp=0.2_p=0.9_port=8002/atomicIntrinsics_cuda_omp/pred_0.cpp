Here's the equivalent OpenMP code for the given CUDA code:

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

  T *dOData = (T*)malloc(memSize);

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

    // Call kernel
    #pragma omp parallel for num_threads(numThreads)
    for (int j = 0; j < numBlocks; j++) {
      #pragma omp task
      {
        testKernel<T>(dOData + j * numThreads);
      }
    }
    #pragma omp taskwait

    // Compute gold
    computeGold<T>(dOData, numThreads * numBlocks);

    auto start = std::chrono::steady_clock::now();

    // Call kernel again
    #pragma omp parallel for num_threads(numThreads)
    for (int j = 0; j < numBlocks; j++) {
      #pragma omp task
      {
        testKernel<T>(dOData + j * numThreads);
      }
    }
    #pragma omp taskwait

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

    // Copy data back to host
    for (int j = 0; j < numBlocks; j++) {
      #pragma omp parallel for num_threads(numThreads)
      for (int k = 0; k < numThreads; k++) {
        int idx = j * numThreads + k;
        if (idx < len) {
          gpuData[idx] = dOData[idx];
        }
      }
    }
  }

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

Note that the OpenMP code uses parallel for loops to distribute the work among threads, and taskwait to ensure that all tasks are completed before proceeding. The kernel function is called using the task directive, which allows OpenMP to schedule the work among threads. The data is copied to and from the device using parallel for loops.