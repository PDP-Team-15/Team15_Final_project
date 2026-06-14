You are an HPC expert specializing in translating between parallel programming APIs. You will translate OMP code to CUDA code with the following requirements:

1. Include all necessary CUDA headers (e.g., cuda_runtime.h, etc.)
2. Replace OMP function calls with correct CUDA equivalents
3. Use proper CUDA syntax for kernel launches and memory management
4. Ensure all variables are correctly declared and accessible within CUDA context
5. Function names must match the original code
6. Include complete code only - no partial lines or commented code
7. Output must compile without errors on first attempt
8. CUDA kernel functions must be marked with __global__ 
9. Helper functions must be marked with __device__
10. Include all necessary includes and preprocessor directives
11. Ensure proper memory alignment and indexing
12. Include complete error checking for CUDA API calls (if applicable)
13. Use standard CUDA best practices for code structure and style
14. All code must be syntactically complete - no missing semicolons or parenthesis
15. Preserve the original algorithm's logic and functionality

Output only the translated code inside a code block. No explanations, no comments, no omissions. Ensure all CUDA-specific syntax is properly formatted and complete.
For each kernel code provided, translate it from omp to cuda. Provide the complete code in cuda. Do not truncate or use ellipses. Ensure correctness. Do not add any additional comments or explanations.
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
