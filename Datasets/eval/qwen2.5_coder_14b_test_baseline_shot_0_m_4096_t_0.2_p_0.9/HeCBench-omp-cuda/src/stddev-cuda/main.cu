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

__global__ void initStddev(float *std, int D) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < D) {
    std[idx] = 0.0f;
  }
}

__global__ void computeStddev(const float *data, float *std, int D, int N, int sampleSize) {
  extern __shared__ float sstd[];
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int gridDim_x = gridDim.x;

  const int RowsPerBlkPerIter = blockDim.x / blockDim.y;
  int thisColId = tx % blockDim.y;
  int thisRowId = tx / blockDim.y;
  int colId = thisColId + (by * blockDim.y);
  int rowId = thisRowId + (bx * RowsPerBlkPerIter);
  float thread_data = 0.0f;
  const int stride = RowsPerBlkPerIter * gridDim_x;
  for (int i = rowId; i < N; i += stride) {
    float val = (colId < D) ? data[i * D + colId] : 0.0f;
    thread_data += val * val;
  }
  if (tx < blockDim.y) sstd[tx] = 0.0f;
  __syncthreads();

  atomicAdd(&sstd[thisColId], thread_data);

  __syncthreads();

  if (tx < blockDim.y) {
    atomicAdd(&std[colId], sstd[thisColId]);
  }
}

template <typename Type, typename IdxType = int>
void stddev(Type *std, const Type *data, IdxType D, IdxType N, bool sample) {
  static const int TPB = 256;
  static const int RowsPerThread = 4;
  static const int ColsPerBlk = 32;
  static const int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;

  static const int TeamX = (N + (IdxType)RowsPerBlk - 1) / (IdxType)RowsPerBlk;
  static const int TeamY = (D + (IdxType)ColsPerBlk - 1) / (IdxType)ColsPerBlk;
  static const int Teams = TeamX * TeamY;

  dim3 block(ColsPerBlk, RowsPerBlk / RowsPerThread);
  dim3 grid(TeamX, TeamY);

  initStddev<<<grid, block>>>(std, D);

  computeStddev<<<grid, block, ColsPerBlk * sizeof(float)>>>(data, std, D, N, sample ? N - 1 : N);

  cudaDeviceSynchronize();
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
  for (int i = 0; i < N; i++)
    for (int j = 0; j < D; j++)
      data[i * D + j] = rand() / (float)RAND_MAX;

  long outputSize = D;
  long outputSizeByte = outputSize * sizeof(float);
  float *std = (float*)malloc(outputSizeByte);
  float *std_ref = (float*)malloc(outputSizeByte);

  float *d_data, *d_std;
  cudaMalloc(&d_data, inputSizeByte);
  cudaMalloc(&d_std, outputSizeByte);

  cudaMemcpy(d_data, data, inputSizeByte, cudaMemcpyHostToDevice);

  stddev(d_std, d_data, D, N, sample);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    stddev(d_std, d_data, D, N, sample);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of stddev kernels: %f (s)\n", (time * 1e-9f) / repeat);

  cudaMemcpy(std, d_std, outputSizeByte, cudaMemcpyDeviceToHost);

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
  cudaFree(d_data);
  cudaFree(d_std);

  return 0;
}