#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "reference.h"

template <class T>
void testcase(const int repeat)
{
    const int len = 1 << 10;
    unsigned int numThreads = 256;
    unsigned int numData = 7;
    unsigned int memSize = sizeof(T) * numData;
    const T data[] = {0, 0, (T)-256, 256, 255, 0, 255};
    T* gpuData;
    cudaMalloc((void**)&gpuData, memSize);

    cudaMemcpy(gpuData, data, memSize, cudaMemcpyHostToDevice);

    computeGold<T>(gpuData, len);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    for (int n = 0; n < repeat; n++) {
        dim3 threads(numThreads);
        dim3 blocks(1);

        kernel_atomic<T><<<blocks, threads>>>(gpuData, len);
        kernel_reduction<T><<<blocks, threads>>>(gpuData, len);
    }

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float time;
    cudaEventElapsedTime(&time, startEvent, stopEvent);

    printf("Average kernel execution time: %f (us)\n", time / repeat);

    cudaMemcpy(gpuData, data, memSize, cudaMemcpyHostToDevice);
    computeGold<T>(gpuData, len);

    cudaFree(gpuData);
}

__global__ void kernel_atomic(T* gpuData, int len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        atomicAdd(gpuData, 10);
        atomicSub(gpuData + 1, 10);
        atomicAnd(gpuData + 4, (T)(2 * i + 7));
        atomicOr(gpuData + 5, (T)(1 << i));
        atomicXor(gpuData + 6, (T)i);
    }
}

__global__ void kernel_reduction(T* gpuData, int len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        atomicMax(gpuData + 2, (T)i);
        atomicMin(gpuData + 3, (T)i);
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