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

  dim3 initGrid(D);
  initKernel<Type><<<initGrid, TPB>>>(std, D);

  dim3 grid(TeamX, TeamY);
  dim3 block(TPB);
  
  computeKernel<Type, IdxType><<<grid, block>>>(std, data, D, N, ColsPerBlk, RowsPerBlk, TeamX, TeamY);

  IdxType sampleSize = sample ? N-1 : N;
  dim3 finalizeGrid(D);
  finalizeKernel<Type><<<finalizeGrid, TPB>>>(std, D, sampleSize);
}

template <typename Type>
__global__ void initKernel(Type *std, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < D) {
    std[idx] = Type(0);
  }
}

template <typename Type, typename IdxType>
__global__ void computeKernel(Type *std, const Type *data, IdxType D, IdxType N, 
                             int ColsPerBlk, int RowsPerBlk, int TeamX, int TeamY) {
  __shared__ Type sstd[ColsPerBlk];
  
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  
  int thisColId = tx % ColsPerBlk;
  int thisRowId = tx / ColsPerBlk;
  
  int colId = thisColId + (by * ColsPerBlk);
  int rowId = thisRowId + (bx * (TPB / ColsPerBlk));
  
  Type thread_data = Type(0);
  int stride = (TPB / ColsPerBlk) * TeamX;
  
  for (IdxType i = rowId; i < N; i += stride) {
    if (colId < D) {
      Type val = data[i * D + colId];
      thread_data += val * val;
    }
  }
  
  sstd[thisColId] = thread_data;
  
  __syncthreads();
  
  if (thisColId < ColsPerBlk) {
    atomicAdd(&std[colId], sstd[thisColId]);
  }
}

template <typename Type>
__global__ void finalizeKernel(Type *std, int D, int sampleSize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < D) {
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
  
  float *data;
  cudaMalloc(&data, inputSizeByte);
  
  srand(123);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < D; j++) {
      data[i*D + j] = rand() / (float)RAND_MAX; 
    }
  }
  
  long outputSize = D;
  long outputSizeByte = outputSize * sizeof(float);
  
  float *std, *std_ref;
  cudaMalloc(&std, outputSizeByte);
  cudaMallocHost(&std_ref, outputSizeByte);
  
  cudaGetLastError();
  
  for (int i = 0; i < repeat; i++) {
    stddev(std, data, D, N, sample);
  }
  
  cudaMemcpy(std_ref, std, outputSizeByte, cudaMemcpyDeviceToHost);
  
  bool ok = true;
  for (int i = 0; i < D; i++) {
    if (fabsf(std_ref[i] - std[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  
  printf("%s\n", ok ? "PASS" : "FAIL");
  
  cudaFree(std_ref);
  cudaFree(std);
  cudaFree(data);
  
  return 0;
}