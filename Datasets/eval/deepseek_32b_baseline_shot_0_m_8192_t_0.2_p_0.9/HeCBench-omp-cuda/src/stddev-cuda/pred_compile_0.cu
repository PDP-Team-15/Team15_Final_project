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

template <typename Type>
__global__ void init_kernel(Type *std, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < D) {
        std[idx] = 0;
    }
}

template <typename Type, int ColsPerBlk>
__global__ void compute_kernel(Type *std, const Type *data, int D, int N, int RowsPerBlk, int TeamX, int TeamY) {
    __shared__ Type sstd[ColsPerBlk];
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int thisColId = tx % ColsPerBlk;
    int thisRowId = tx / ColsPerBlk;
    int colId = thisColId + by * ColsPerBlk;
    int rowId = thisRowId + bx * (RowsPerBlk / ColsPerBlk);

    Type thread_data = 0;
    int stride = (RowsPerBlk / ColsPerBlk) * TeamX;

    for (int i = rowId; i < N; i += stride) {
        if (colId < D) {
            int data_idx = i * D + colId;
            Type val = data[data_idx];
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
__global__ void finalize_kernel(Type *std, int D, bool sample, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < D) {
        int sampleSize = sample ? N-1 : N;
        std[idx] = sqrtf(std[idx] / sampleSize);
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

    // Initialize std array
    dim3 init_grid(D);
    init_kernel<Type><<<init_grid, TPB>>>(std, D);

    // Main computation kernel
    dim3 grid(TeamX, TeamY);
    dim3 block(TPB);
    compute_kernel<Type, ColsPerBlk><<<grid, block>>>(std, data, D, N, RowsPerBlk, TeamX, TeamY);

    // Finalize std array
    dim3 finalize_grid(D);
    finalize_kernel<Type><<<finalize_grid, TPB>>>(std, D, sample, N);
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <D> <N> <repeat>\n", argv[0]);
    printf("D: number of columns of data (must be a multiple of 32)\n");
    printf("N: number of rows of data (at least one row)\n");
    return 1;
  }
  int D = atoi(argv[1]); // columns must be a multiple of 32
  int N = atoi(argv[2]); // at least one row
  int repeat = atoi(argv[3]);

  bool sample = true;
  long inputSize = D * N;
  long inputSizeByte = inputSize * sizeof(float);
  float *data = (float*) malloc (inputSizeByte);

  // input data 
  srand(123);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < D; j++) 
      data[i*D + j] = rand() / (float)RAND_MAX; 

  float *d_data;
  cudaMalloc((void**)&d_data, inputSizeByte);
  cudaMemcpy(d_data, data, inputSizeByte, cudaMemcpyHostToDevice);

  // host and device results
  long outputSize = D;
  long outputSizeByte = outputSize * sizeof(float);
  float *std  = (float*) malloc (outputSizeByte);
  float *std_ref  = (float*) malloc (outputSizeByte);
  float *d_std;
  cudaMalloc((void**)&d_std, outputSizeByte);

  // warmup
  stddev(d_std, d_data, D, N, sample);

  // execute kernels on a device
  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    stddev(d_std, d_data, D, N, sample);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of stddev kernels: %f (s)\n", (time * 1e-9f) / repeat);

  cudaMemcpy(std, d_std, outputSizeByte, cudaMemcpyDeviceToHost);

  // verify
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

Error code: 400 - {'object': 'error', 'message': "This model's maximum context length is 16384 tokens. However, you requested 16530 tokens (1530 in the messages, 15000 in the completion). Please reduce the length of the messages or completion.", 'type': 'BadRequestError', 'param': None, 'code': 400}