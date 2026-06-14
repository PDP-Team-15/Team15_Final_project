To translate the given OpenMP code to CUDA, we need to replace the OpenMP constructs with their CUDA equivalents. Below is the complete CUDA version of the code:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda_runtime.h>

#include "reference.h"

template <class T>
__global__ void kernel(T* gpuData, int len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len)
    {
        atomicAdd(&gpuData[0], (T)10);
        atomicSub(&gpuData[1], (T)10);

        atomicAnd(&gpuData[4], (T)(2 * i + 7));
        atomicOr(&gpuData[5], (T)(1 << i));
        atomicXor(&gpuData[6], (T)i);
    }
}

template <class T>
__global__ void maxKernel(T* gpuData, int len)
{
    extern __shared__ T sdata[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len)
    {
        sdata[threadIdx.x] = gpuData[2];
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            sdata[threadIdx.x] = max(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        gpuData[2] = sdata[0];
    }
}

template <class T>
__global__ void minKernel(T* gpuData, int len)
{
    extern __shared__ T sdata[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len)
    {
        sdata[threadIdx.x] = gpuData[3];
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            sdata[threadIdx.x] = min(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        gpuData[3] = sdata[0];
    }
}

template <class T>
void testcase(const int repeat)
{
    const int len = 1 << 10;
    unsigned int numThreads = 256;
    unsigned int numBlocks = (len + numThreads - 1) / numThreads;
    unsigned int numData = 7;
    unsigned int memSize = sizeof(T) * numData;
    const T data[] = {0, 0, (T)-256, 256, 255, 0, 255};
    T* gpuData;

    cudaMalloc(&gpuData, memSize);
    cudaMemcpy(gpuData, data, memSize, cudaMemcpyHostToDevice);

    for (int n = 0; n < repeat; n++)
    {
        kernel<<<numBlocks, numThreads>>>(gpuData, len);
        maxKernel<<<numBlocks, numThreads, numThreads * sizeof(T)>>>(gpuData, len);
        minKernel<<<numBlocks, numThreads, numThreads * sizeof(T)>>>(gpuData, len);
    }

    cudaMemcpy(gpuData, data, memSize, cudaMemcpyHostToDevice);

    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++)
    {
        kernel<<<numBlocks, numThreads>>>(gpuData, len);
        maxKernel<<<numBlocks, numThreads, numThreads * sizeof(T)>>>(gpuData, len);
        minKernel<<<numBlocks, numThreads, numThreads * sizeof(T)>>>(gpuData, len);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

    computeGold<T>(gpuData, len);

    cudaFree(gpuData);
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

### Key Changes:
1. **Kernel Functions**: Replaced OpenMP parallel for loops with CUDA kernel functions.
2. **Atomic Operations**: Used `atomicAdd`, `atomicSub`, `atomicAnd`, `atomicOr`, and `atomicXor` for atomic operations.
3. **Reduction Operations**: Implemented custom reduction kernels using shared memory for `max` and `min` operations.
4. **Memory Management**: Used `cudaMalloc` and `cudaMemcpy` for device memory allocation and data transfer.
5. **Launch Configuration**: Configured the number of blocks and threads for kernel launches.

This CUDA code should perform the same operations as the original OpenMP code but using CUDA parallelism.