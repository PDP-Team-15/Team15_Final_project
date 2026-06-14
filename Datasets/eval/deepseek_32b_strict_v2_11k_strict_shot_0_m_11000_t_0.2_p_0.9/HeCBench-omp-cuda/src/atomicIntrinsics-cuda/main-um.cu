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
#include <cuda.h>
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
    T* gpuDataGold;

    cudaMalloc((void**)&gpuData, memSize);
    cudaMemcpy(gpuData, data, memSize, cudaMemcpyHostToDevice);

    dim3 threads(numThreads);
    dim3 blocks(1);

    for (int n = 0; n < repeat; n++) {
        cudaMemcpy(gpuData, data, memSize, cudaMemcpyHostToDevice);

        kernel_atomic<<<blocks, threads>>>(gpuData, len);
        kernel_reduction<<<blocks, threads>>>(gpuData, len, T(0));
        kernel_reduction<<<blocks, threads>>>(gpuData, len, T(255));

        cudaMemcpy(gpuDataGold, gpuData, memSize, cudaMemcpyDeviceToHost);
        computeGold<T>(gpuDataGold, len);
    }

    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
        kernel_atomic<<<blocks, threads>>>(gpuData, len);
        kernel_reduction<<<blocks, threads>>>(gpuData, len, T(0));
        kernel_reduction<<<blocks, threads>>>(gpuData, len, T(255));
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

    cudaFree(gpuData);
}

__global__ void kernel_atomic(T* gpuData, int len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        atomicAdd((T*)&gpuData[0], (T)10);
        atomicSub((T*)&gpuData[1], (T)10);
        atomicAnd((T*)&gpuData[4], (T)(2 * i + 7));
        atomicOr((T*)&gpuData[5], (T)(1 << i));
        atomicXor((T*)&gpuData[6], (T)i);
    }
}

__global__ void kernel_reduction(T* gpuData, int len, T initialValue)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        if (gpuData[2] < (T)i) {
            atomicMax((T*)&gpuData[2], (T)i);
        }
        if (gpuData[3] > (T)i) {
            atomicMin((T*)&gpuData[3], (T)i);
        }
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