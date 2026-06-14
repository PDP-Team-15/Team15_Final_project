Here is the equivalent CUDA code for the given OpenMP code:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda_runtime.h>

#include "reference.h"

template <class T>
__global__ void kernel(const T data[], const int len, T gpuData[], const int numData) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    atomicAdd(&gpuData[0], (T)10);
    atomicSub(&gpuData[1], (T)10);
    gpuData[4] = gpuData[4] & (T)(2 * idx + 7);
    gpuData[5] = gpuData[5] | (T)(1 << idx);
    gpuData[6] = gpuData[6] ^ (T)idx;
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    gpuData[2] = fmax(gpuData[2], (T)idx);
    gpuData[3] = fmin(gpuData[3], (T)idx);
  }

  __syncthreads();
}

template <class T>
void testcase(const int repeat) {
  const int len = 1 << 10;
  const int numThreads = 256;
  const int numData = 7;
  const T data[] = {0, 0, (T)-256, 256, 255, 0, 255};
  T gpuData[7];

  cudaMalloc((void**)&gpuData, sizeof(T) * numData);

  for (int n = 0; n < repeat; n++) {
    cudaMemcpy(gpuData, data, sizeof(T) * numData, cudaMemcpyHostToDevice);

    kernel<T><<<1, numThreads>>>(data, len, gpuData, numData);
    cudaDeviceSynchronize();

    kernel<T><<<1, numThreads>>>(data, len, gpuData, numData);
    cudaDeviceSynchronize();

    computeGold<T>(gpuData, len);

    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      kernel<T><<<1, numThreads>>>(data, len, gpuData, numData);
      cudaDeviceSynchronize();

      kernel<T><<<1, numThreads>>>(data, len, gpuData, numData);
      cudaDeviceSynchronize();
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);
  }

  cudaMemcpy(data, gpuData, sizeof(T) * numData, cudaMemcpyDeviceToHost);
  cudaFree(gpuData);
}

int main(int argc, char **argv) {
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

Note that I've made the following changes:

* Replaced OpenMP pragmas with CUDA kernel calls.
* Replaced OpenMP atomic updates with CUDA atomicAdd and atomicSub functions.
* Replaced OpenMP reduction with CUDA kernel synchronization using __syncthreads.
* Replaced OpenMP target data map with CUDA memory allocation using cudaMalloc.
* Replaced OpenMP target update to and from with CUDA memory copy using cudaMemcpy.
* Replaced OpenMP target teams distribute parallel for with CUDA kernel launch using <<< >>>.
* Replaced OpenMP thread_limit with CUDA block size.
* Replaced OpenMP reduction max and min with CUDA fmax and fmin functions.
* Removed OpenMP parallel for loop and replaced it with a single kernel launch.
* Removed OpenMP atomic update for gpuData[2] and gpuData[3] and replaced it with a single kernel launch.
* Removed OpenMP atomic update for gpuData[4], gpuData[5], and gpuData[6] and replaced it with a single kernel launch.