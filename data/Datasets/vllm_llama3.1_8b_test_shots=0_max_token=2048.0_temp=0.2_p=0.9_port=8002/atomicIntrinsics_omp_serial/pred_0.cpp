Here's the serial version of the provided code:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

#include "reference.h"

template <class T>
void testcase(const int repeat)
{
  const int len = 1 << 10;
  unsigned int numData = 7;
  unsigned int memSize = sizeof(T) * numData;
  const T data[] = {0, 0, (T)-256, 256, 255, 0, 255};
  T gpuData[7];

  for (int n = 0; n < repeat; n++) {
    memcpy(gpuData, data, memSize);

    for (int i = 0; i < len; ++i)
    {
      gpuData[0] += (T)10;
      gpuData[1] -= (T)10;
      gpuData[4] &= (T)(2*i+7);
      gpuData[5] |= (T)(1 << i);
      gpuData[6] ^= (T)i;
    }

    for (int i = 0; i < len; ++i)
      gpuData[2] = std::max(gpuData[2], (T)i);

    for (int i = 0; i < len; ++i)
      gpuData[3] = std::min(gpuData[3], (T)i);

    computeGold<T>(gpuData, len);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < len; ++i)
    {
      gpuData[0] += (T)10;
      gpuData[1] -= (T)10;
      gpuData[4] &= (T)(2*i+7);
      gpuData[5] |= (T)(1 << i);
      gpuData[6] ^= (T)i;
    }

    for (int i = 0; i < len; ++i)
      gpuData[2] = std::max(gpuData[2], (T)i);

    for (int i = 0; i < len; ++i)
      gpuData[3] = std::min(gpuData[3], (T)i);
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

Note that the `#pragma omp target` directives and the `#pragma omp atomic update` directives have been removed, as they are specific to OpenMP parallel programming and are not applicable in a serial version of the code. The `std::max` and `std::min` functions have been used instead of the `#pragma omp target teams distribute parallel for thread_limit(256) reduction(max/min: ...)` directives to compute the maximum and minimum values.