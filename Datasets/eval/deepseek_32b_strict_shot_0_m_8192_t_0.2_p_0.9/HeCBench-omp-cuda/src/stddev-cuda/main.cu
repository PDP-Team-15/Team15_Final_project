#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>
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
  dim3 init_blocks(D / 256 + 1);
  dim3 init_threads(256);
  init_std<<<init_blocks, init_threads>>>(std, D);

  // Main computation
  dim3 blocks(TeamX, TeamY);
  dim3 threads(TPB);
  main_kernel<<<blocks, threads>>>(std, data, D, N, ColsPerBlk, RowsPerBlk, TPB);

  // Compute square root
  dim3 sqrt_blocks(D / 256 + 1);
  dim3 sqrt_threads(256);
  compute_sqrt<<<sqrt_blocks, sqrt_threads>>>(std, D, sample);
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
  float *data = (float*) malloc (inputSizeByte);

  srand(123);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < D; j++) 
      data[i*D + j] = rand() / (float)RAND_MAX; 

  long outputSize = D;
  long outputSizeByte = outputSize * sizeof(float);
  float *std  = (float*) malloc (outputSizeByte);
  float *std_ref  = (float*) malloc (outputSizeByte);

  cudaMalloc((void**)&data, inputSizeByte);
  cudaMalloc((void**)&std, outputSizeByte);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    stddev(std, data, D, N, sample);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of stddev kernels: %f (s)\n", (time * 1e-9f) / repeat);

  cudaMemcpy(std_ref, std, outputSizeByte, cudaMemcpyDeviceToHost);

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
  return 0;
}

// CUDA Kernels
__global__ void init_std(float *std, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < D) std[idx] = 0.0f;
}

__global__ void main_kernel(float *std, const float *data, int D, int N, 
                           int ColsPerBlk, int RowsPerBlk, int TPB) {
  __shared__ float sstd[ColsPerBlk];
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int thisColId = tx % ColsPerBlk;
  int thisRowId = tx / ColsPerBlk;
  int colId = thisColId + by * ColsPerBlk;
  int rowId = thisRowId + bx * (TPB / ColsPerBlk);

  float thread_data = 0.0f;
  int stride = (TPB / ColsPerBlk) * gridDim.x;
  for (int i = rowId; i < N; i += stride) {
    if (colId < D) {
      int data_idx = i * D + colId;
      float val = data[data_idx];
      thread_data += val * val;
    }
  }

  sstd[thisColId] = thread_data;
  __syncthreads();

  if (thisColId < ColsPerBlk && colId < D) {
    atomicAdd(&std[colId], sstd[thisColId]);
  }
}

__global__ void compute_sqrt(float *std, int D, bool sample) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < D) {
    float sampleSize = sample ? (D - 1) : D;
    std[idx] = sqrtf(std[idx] / sampleSize);
  }
}