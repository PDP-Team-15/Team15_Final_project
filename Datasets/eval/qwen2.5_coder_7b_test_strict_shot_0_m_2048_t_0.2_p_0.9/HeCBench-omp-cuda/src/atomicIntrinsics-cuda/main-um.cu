#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda_runtime.h>
#include "reference.h"

template <class T>
__global__ void kernel(T* gpuData, int len, int numThreads) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < len) {
        atomicAdd(&gpuData[0], (T)10);
        atomicSub(&gpuData[1], (T)10);
        atomicAnd(&gpuData[4], (T)(2*i+7));
        atomicOr(&gpuData[5], (T)(1 << i));
        atomicXor(&gpuData[6], (T)i);
    }

    __shared__ T sharedMax[256];
    sharedMax[threadIdx.x] = (threadIdx.x == 0) ? gpuData[2] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedMax[threadIdx.x] = max(sharedMax[threadIdx.x], sharedMax[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMax(&gpuData[2], sharedMax[0]);
    }

    __shared__ T sharedMin[256];
    sharedMin[threadIdx.x] = (threadIdx.x == 0) ? gpuData[3] : len;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedMin[threadIdx.x] = min(sharedMin[threadIdx.x], sharedMin[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMin(&gpuData[3], sharedMin[0]);
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

    cudaMemcpy(gpuData, data, memSize, cudaMemcpyHostToDevice);

    for (int n = 0; n < repeat; n++) {
        kernel<<<len / numThreads, numThreads>>>(gpuData, len, numThreads);
    }

    cudaMemcpy(data, gpuData, memSize, cudaMemcpyDeviceToHost);

    computeGold<T>(data, len);

    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
        kernel<<<len / numThreads, numThreads>>>(gpuData, len, numThreads);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);
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