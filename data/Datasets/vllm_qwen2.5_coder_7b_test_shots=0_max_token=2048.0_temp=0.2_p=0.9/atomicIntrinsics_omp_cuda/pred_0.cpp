Here is the translation of the provided OpenMP code to CUDA:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda_runtime.h>
#include "reference.h"

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

template <class T>
__global__ void kernel(T* gpuData, int len, int numThreads) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < len; i += stride) {
        atomicAdd(&gpuData[0], static_cast<T>(10));
        atomicSubtract(&gpuData[1], static_cast<T>(10));

        atomicAnd(&gpuData[4], static_cast<T>(2 * i + 7));
        atomicOr(&gpuData[5], static_cast<T>(1 << i));
        atomicXor(&gpuData[6], static_cast<T>(i));
    }

    __shared__ T sharedMax, sharedMin;
    if (tid == 0) {
        sharedMax = gpuData[2];
        sharedMin = gpuData[3];
    }
    __syncthreads();

    for (int i = tid; i < len; i += stride) {
        sharedMax = max(sharedMax, static_cast<T>(i));
        sharedMin = min(sharedMin, static_cast<T>(i));
    }
    __syncthreads();

    if (tid == 0) {
        atomicMax(&gpuData[2], sharedMax);
        atomicMin(&gpuData[3], sharedMin);
    }
}

template <class T>
void testcase(const int repeat) {
    const int len = 1 << 10;
    unsigned int numThreads = 256;
    unsigned int numData = 7;
    unsigned int memSize = sizeof(T) * numData;
    const T data[] = {0, 0, (T)-256, 256, 255, 0, 255};
    T gpuData[7];

    T* d_gpuData;
    CHECK_CUDA(cudaMalloc((void**)&d_gpuData, memSize));

    for (int n = 0; n < repeat; n++) {
        memcpy(gpuData, data, memSize);
        CHECK_CUDA(cudaMemcpy(d_gpuData, gpuData, memSize, cudaMemcpyHostToDevice));

        dim3 blocks(len / numThreads + (len % numThreads ? 1 : 0), 1, 1);
        dim3 threads(numThreads, 1, 1);

        kernel<<<blocks, threads>>>(d_gpuData, len, numThreads);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(gpuData, d_gpuData, memSize, cudaMemcpyDeviceToHost));
    }

    computeGold<T>(gpuData, len);

    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
        memcpy(gpuData, data, memSize);
        CHECK_CUDA(cudaMemcpy(d_gpuData, gpuData, memSize, cudaMemcpyHostToDevice));

        kernel<<<blocks, threads>>>(d_gpuData, len, numThreads);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

    CHECK_CUDA(cudaFree(d_gpuData));
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

### Explanation:
1. **Kernel Function (`kernel`)**:
   - The kernel function is defined with `__global__` and runs on the GPU.
   - It performs the same operations as the original OpenMP code but using CUDA's atomic functions.

2. **Memory Management**:
   - Memory allocation and deallocation are handled using `cudaMalloc` and `cudaFree`.
   - Data transfer between host and device is done using `cudaMemcpy`.

3. **Execution**:
   - The kernel is launched with appropriate block and grid dimensions.
   - Synchronization is ensured using `cudaDeviceSynchronize`.

4. **Timing**:
   - Timing is performed similarly to the original code, measuring the average execution time over multiple repetitions.

This CUDA version should provide equivalent functionality to the original OpenMP code.