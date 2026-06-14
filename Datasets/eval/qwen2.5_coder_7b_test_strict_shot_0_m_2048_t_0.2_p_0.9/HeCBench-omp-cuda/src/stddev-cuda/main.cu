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

#define TPB 256
#define RowsPerThread 4
#define ColsPerBlk 32
#define RowsPerBlk ((TPB / ColsPerBlk) * RowsPerThread)

__global__ void initStddev(float *std, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < D) {
    std[idx] = 0.0f;
  }
}

__global__ void computeStddev(float *std, const float *data, int D, int N, bool sample) {
  extern __shared__ float sstd[];
  int tx = threadIdx.x;
  int bx = blockIdx.x % gridDim.x;
  int by = blockIdx.x / gridDim.x;
  int gridDim_x = gridDim.x;

  const int RowsPerBlkPerIter = TPB / ColsPerBlk;
  int thisColId = tx % ColsPerBlk;
  int thisRowId = tx / ColsPerBlk;
  int colId = thisColId + ((int)by * ColsPerBlk);
  int rowId = thisRowId + ((int)bx * RowsPerBlkPerIter);
  float thread_data = 0.0f;
  const int stride = RowsPerBlkPerIter * gridDim_x;
  for (int i = rowId; i < N; i += stride) {
    float val = (colId < D) ? data[i * D + colId] : 0.0f;
    thread_data += val * val;
  }
  if (tx < ColsPerBlk) sstd[tx] = 0.0f;
  __syncthreads();

  atomicAdd(&sstd[thisColId], thread_data);
  __syncthreads();

  if (tx < ColsPerBlk) {
    atomicAdd(&std[colId], sstd[thisColId]);
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
  srand(123);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < D; j++) 
      data[i*D + j] = rand() / (float)RAND_MAX; 

  long outputSize = D;
  long outputSizeByte = outputSize * sizeof(float);
  float *std  = (float*) malloc(outputSizeByte);
  float *std_ref  = (float*) malloc(outputSizeByte);

  float *d_std, *d_data;
  cudaMalloc((void**)&d_std, outputSizeByte);
  cudaMalloc((void**)&d_data, inputSizeByte);

  cudaMemcpy(d_data, data, inputSizeByte, cudaMemcpyHostToDevice);

  dim3 blockSize(TPB);
  dim3 gridSize(((D + ColsPerBlk - 1) / ColsPerBlk), ((N + RowsPerBlk - 1) / RowsPerBlk));

  initStddev<<<gridSize, blockSize>>>(d_std, D);
  cudaDeviceSynchronize();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    computeStddev<<<gridSize, blockSize, ColsPerBlk * sizeof(float)>>>(d_std, d_data, D, N, sample);
    cudaDeviceSynchronize();
  }

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
  cudaFree(d_std);
  cudaFree(d_data);
  return 0;
}