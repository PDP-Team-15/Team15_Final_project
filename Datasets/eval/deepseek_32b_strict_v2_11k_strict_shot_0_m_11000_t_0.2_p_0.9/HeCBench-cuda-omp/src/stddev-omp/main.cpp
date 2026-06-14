#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "reference.h"

template <typename Type, typename IdxType>
void sampleKernelOMP(Type *std, IdxType D, IdxType N) {
    #pragma omp parallel for
    for (IdxType i = 0; i < D; ++i) {
        std[i] = sqrtf(std[i] / N);
    }
}

template <typename Type, typename IdxType>
void sopKernelOMP(Type *std, const Type *data, IdxType D, IdxType N) {
    const int TPB = 256;
    const int ColsPerBlk = 32;
    const int RowsPerBlk = (TPB / ColsPerBlk) * 4;
    
    int numBlocksX = (N + RowsPerBlk - 1) / RowsPerBlk;
    int numBlocksY = (D + ColsPerBlk - 1) / ColsPerBlk;
    
    Type *sstd = new Type[ColsPerBlk];
    #pragma omp parallel
    {
        #pragma omp for
        for (int blockY = 0; blockY < numBlocksY; ++blockY) {
            for (int blockX = 0; blockX < numBlocksX; ++blockX) {
                int colId = blockY * ColsPerBlk;
                int rowId = blockX * RowsPerBlk;
                
                Type thread_data = 0;
                for (int i = rowId; i < N; i += RowsPerBlk * numBlocksX) {
                    for (int j = 0; j < RowsPerBlk; ++j) {
                        int currentRow = i + j;
                        if (currentRow < N) {
                            for (int k = 0; k < ColsPerBlk; ++k) {
                                int currentCol = colId + k;
                                if (currentCol < D) {
                                    Type val = data[currentRow * D + currentCol];
                                    thread_data += val * val;
                                }
                            }
                        }
                    }
                }
                
                int threadId = omp_get_thread_num();
                #pragma omp atomic
                sstd[threadId] += thread_data;
                
                #pragma omp barrier
                
                if (threadId < ColsPerBlk) {
                    #pragma omp atomic
                    std[colId + threadId] += sstd[threadId];
                }
            }
        }
    }
    delete[] sstd;
}

template <typename Type, typename IdxType = int>
void stddevOMP(Type *std, const Type *data, IdxType D, IdxType N, bool sample) {
    #pragma omp parallel
    {
        #pragma omp for
        for (IdxType i = 0; i < D; ++i) {
            std[i] = 0;
        }
    }
    
    sopKernelOMP(std, data, D, N);
    
    IdxType sampleSize = sample ? N - 1 : N;
    sampleKernelOMP(std, D, sampleSize);
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
    float *data = (float*)malloc(inputSizeByte);
    
    srand(123);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            data[i * D + j] = rand() / (float)RAND_MAX;
        }
    }
    
    float *std = (float*)malloc(D * sizeof(float));
    float *std_ref = (float*)malloc(D * sizeof(float));
    
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
        stddevOMP(std, data, D, N, sample);
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of stddev kernels: %f (s)\n", (time * 1e-9f) / repeat);
    
    stddev_ref(std_ref, data, D, N, sample);
    
    bool ok = true;
    for (int i = 0; i < D; ++i) {
        if (fabsf(std_ref[i] - std[i]) > 1e-3) {
            ok = false;
            break;
        }
    }
    
    printf("%s\n", ok ? "PASS" : "FAIL");
    free(std_ref);
    free(std);
    free(data);
    return 0;
}