#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "reference.h"

template <typename Type, typename IdxType = int>
void stddev(Type *std, const Type *data, IdxType D, IdxType N, bool sample) {
    static const int TPB = 256;
    static const int RowsPerThread = 4;
    static const int ColsPerBlk = 32;
    static const int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;

    static const int TeamX = (N + (IdxType)RowsPerBlk - 1) / (IdxType)RowsPerBlk;
    static const int TeamY = (D + (IdxType)ColsPerBlk - 1) / (IdxType)ColsPerBlk;
    static const int Teams = TeamX * TeamY;

    // Initialize std to zero
    dim3 initGrid(D);
    initStdKernel<Type><<<initGrid, TPB>>>(std);
    checkCudaErrors(cudaGetLastError());

    // Compute sum of squares
    dim3 grid(TeamX, TeamY);
    dim3 block(TPB);
    computeSumKernel<Type, IdxType><<<grid, block>>>(std, data, D, N, ColsPerBlk, RowsPerBlk, TeamX, TeamY);
    checkCudaErrors(cudaGetLastError());

    // Finalize std calculation
    dim3 finalizeGrid(D);
    finalizeStdKernel<Type><<<finalizeGrid, TPB>>>(std, D, N, sample);
    checkCudaErrors(cudaGetLastError());
}

__global__ void initStdKernel(Type *std) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < D) {
        std[idx] = 0;
    }
}

__global__ void computeSumKernel(Type *std, const Type *data, IdxType D, IdxType N, int ColsPerBlk, int RowsPerBlk, int TeamX, int TeamY) {
    __shared__ Type sstd[ColsPerBlk];
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    IdxType colId = tx % ColsPerBlk + (IdxType)by * ColsPerBlk;
    IdxType rowId = (tx / ColsPerBlk) + (IdxType)bx * (TPB / ColsPerBlk);

    Type thread_data = 0;
    const IdxType stride = (TPB / ColsPerBlk) * TeamX;
    for (IdxType i = rowId; i < N; i += stride) {
        if (colId < D) {
            Type val = data[i * D + colId];
            thread_data += val * val;
        }
    }

    sstd[tx % ColsPerBlk] = thread_data;
    __syncthreads();

    if (tx % ColsPerBlk == 0) {
        atomicAdd(&std[colId], sstd[tx % ColsPerBlk]);
    }
}

__global__ void finalizeStdKernel(Type *std, IdxType D, IdxType N, bool sample) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < D) {
        IdxType sampleSize = sample ? N - 1 : N;
        std[idx] = sqrtf(std[idx] / sampleSize);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: %s <D> <N> <repeat>\n", argv[0]);
        printf("D: number of columns of data (must be a multiple of 32)\n");
        printf("N: number of rows of data (at least one row)\n");
        return 1;
    }
    int D = atoi(argv[1]); 
    int N = atoi(argv[2]); 
    int repeat = atoi(argv[3]);

    bool sample = true;
    long inputSize = D * N;
    long inputSizeByte = inputSize * sizeof(float);
    float *data = (float*) malloc(inputSizeByte);
    float *std = (float*) malloc(inputSizeByte);
    float *std_ref = (float*) malloc(inputSizeByte);

    srand(123);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++) 
            data[i*D + j] = rand() / (float)RAND_MAX; 

    float *d_data, *d_std;
    checkCudaErrors(cudaMalloc(&d_data, inputSizeByte));
    checkCudaErrors(cudaMalloc(&d_std, inputSizeByte));
    checkCudaErrors(cudaMemcpy(d_data, data, inputSizeByte, cudaMemcpyHostToDevice));

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
        stddev(d_std, d_data, D, N, sample);
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of stddev kernels: %f (s)\n", (time * 1e-9f) / repeat);

    checkCudaErrors(cudaMemcpy(std, d_std, inputSizeByte, cudaMemcpyDeviceToHost));

    stddev_ref(std_ref, data, D, N, sample);

    bool ok = true;
    for (int i = 0; i < D; i++) {
        if (fabsf(std_ref[i] - std[i]) > 1e-3) {
            ok = false;
            break;
        }
    }

    printf("%s\n", ok ? "PASS" : "FAIL");
    free(std_ref);
    free(std);
    free(data);
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_std));
    return 0;
}

void checkCudaErrors(cudaError_t error) {
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}