Here are the serial versions of the provided CUDA code:

**serial.c**

```c
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
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

  for (int i = 0; i < repeat; i++) {
    // No need to copy data to device, we're running on the host
    testKernel<T>(gpuData);
  }

  computeGold<T>(gpuData, numThreads * numBlocks);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    testKernel<T>(gpuData);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);
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

**serial2.c**

```c
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
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

  for (int i = 0; i < repeat; i++) {
    // No need to copy data to device, we're running on the host
    testKernel<T>(gpuData);
  }

  computeGold<T>(gpuData, numThreads * numBlocks);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    testKernel<T>(gpuData);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  // No need to free memory, we're running on the host
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

Note that the main differences between the CUDA code and the serial code are:

*   We no longer need to allocate memory on the device using `cudaMalloc` or `cudaMallocManaged`.
*   We no longer need to copy data to the device using `cudaMemcpy`.
*   We no longer need to synchronize the device using `cudaDeviceSynchronize`.
*   We no longer need to free memory on the device using `cudaFree`.
*   The `testKernel` function now takes a single argument, `gpuData`, which is an array of type `T`.
*   The `computeGold` function now takes two arguments, `gpuData` and `numThreads * numBlocks`, which is the number of threads used in the kernel.

Also, note that the serial code will run much faster than the CUDA code, since it's running on a single core of the CPU.