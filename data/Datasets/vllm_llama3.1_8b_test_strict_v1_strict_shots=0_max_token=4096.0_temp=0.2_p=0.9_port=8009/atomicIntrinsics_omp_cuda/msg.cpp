You are an HPC expert specializing in translating between parallel programming APIs. Translate precisely and completely, preserving all function qualifiers (e.g., const, inline), template parameters, and macro definitions. Output only the translated code inside a code block. No explanations, no comments, no omissions.

When translating to CUDA:
1. Always use proper __host__, __device__, or __global__ qualifiers based on function scope
2. Preserve const qualifiers in function parameters
3. Ensure all #define directives are correctly translated
4. Validate CUDA-specific constructs like memory operations, kernel launches, and stream usage
5. Default to compile-time safety when converting run-time parameters
For each kernel code provided, translate it from omp to cuda. Ensure the translated code is complete, correct, and compiles without errors. Follow these guidelines:

1. Preserve all original functionality and algorithmic logic
2. Replace omp with cuda functions and constructs
3. Correctly declare and initialize all variables
4. Use proper CUDA syntax for kernel launches and memory management
5. Ensure all code is properly terminated with semicolons
6. Include necessary headers
7. Validate code structure and syntax before providing output
Translate the following code  from omp to cuda:
main.cpp:
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

#include <omp.h>
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

  #pragma omp target data map(alloc: gpuData[0:7])
  {
    for (int n = 0; n < repeat; n++) {
      memcpy(gpuData, data, memSize);
      #pragma omp target update to (gpuData[0:7])

      #pragma omp target teams distribute parallel for thread_limit(numThreads)
      for (int i = 0; i < len; ++i)
      {
         #pragma omp atomic update
          gpuData[0] += (T)10;
         #pragma omp atomic update
          gpuData[1] -= (T)10;
         

         

         

         

         #pragma omp atomic update
          gpuData[4] &= (T)(2*i+7);
         #pragma omp atomic update
          gpuData[5] |= (T)(1 << i);
         #pragma omp atomic update
          gpuData[6] ^= (T)i;
      }

      #pragma omp target teams distribute parallel for thread_limit(256) reduction(max: gpuData[2])
      for (int i = 0; i < len; ++i)
         gpuData[2] = max(gpuData[2], (T)i);

      #pragma omp target teams distribute parallel for thread_limit(256) reduction(min: gpuData[3])
      for (int i = 0; i < len; ++i)
         gpuData[3] = min(gpuData[3], (T)i);
    }

    #pragma omp target update from (gpuData[0:7])
    computeGold<T>(gpuData, len);

    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      

      #pragma omp target teams distribute parallel for thread_limit(numThreads)
      for (int i = 0; i < len; ++i)
      {
         #pragma omp atomic update
          gpuData[0] += (T)10;
         #pragma omp atomic update
          gpuData[1] -= (T)10;
         

         

         

         

         #pragma omp atomic update
          gpuData[4] &= (T)(2*i+7);
         #pragma omp atomic update
          gpuData[5] |= (T)(1 << i);
         #pragma omp atomic update
          gpuData[6] ^= (T)i;
      }

      #pragma omp target teams distribute parallel for thread_limit(256) reduction(max: gpuData[2])
      for (int i = 0; i < len; ++i)
         gpuData[2] = max(gpuData[2], (T)i);

      #pragma omp target teams distribute parallel for thread_limit(256) reduction(min: gpuData[3])
      for (int i = 0; i < len; ++i)
         gpuData[3] = min(gpuData[3], (T)i);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);
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
