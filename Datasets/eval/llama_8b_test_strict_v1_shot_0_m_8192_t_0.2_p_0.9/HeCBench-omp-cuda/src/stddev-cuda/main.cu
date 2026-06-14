#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>

#define TPB 256
#define RowsPerThread 4
#define ColsPerBlk 32
#define RowsPerBlk ((TPB / ColsPerBlk) * RowsPerThread)

template <typename Type, typename IdxType = int>
__global__ void stddev_kernel(Type *std, const Type *data, IdxType D, IdxType N, bool sample) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int bx = blockIdx.y;
  int by = blockIdx.z;
  int gridDim_x = gridDim.y;

  const int RowsPerBlkPerIter = TPB / ColsPerBlk;
  IdxType thisColId = tx % ColsPerBlk;
  IdxType thisRowId = tx / ColsPerBlk;
  IdxType colId = thisColId + ((IdxType)by * ColsPerBlk);
  IdxType rowId = thisRowId + ((IdxType)bx * RowsPerBlkPerIter);
  Type thread_data = Type(0);
  const IdxType stride = RowsPerBlkPerIter * gridDim_x;
  for (IdxType i = rowId; i < N; i += stride) {
    Type val = (colId < D) ? data[i * D + colId] : Type(0);
    thread_data += val * val;
  }
  if (tx < ColsPerBlk) std[colId] += thread_data;
}

template <typename Type, typename IdxType = int>
void stddev(Type *std, const Type *data, IdxType D, IdxType N, bool sample) {
  IdxType sampleSize = sample ? N - 1 : N;
  cudaMemset(std, 0, D * sizeof(Type));

  dim3 dimBlock(TPB, 1, 1);
  dim3 dimGrid((D + ColsPerBlk - 1) / ColsPerBlk, 1, 1);
  stddev_kernel<<<dimGrid, dimBlock>>>(std, data, D, N, sample);

  dim3 dimBlock2(ColsPerBlk, 1, 1);
  dim3 dimGrid2((D + ColsPerBlk - 1) / ColsPerBlk, (N + RowsPerBlk - 1) / RowsPerBlk, 1);
  cudaMemsetAsync(std, 0, D * sizeof(Type));
  for (int i = 0; i < (D + ColsPerBlk - 1) / ColsPerBlk; i++) {
    stddev_kernel<<<dimBlock2, dimBlock>>>(std, std, D, N, sample);
    cudaDeviceSynchronize();
  }

  dim3 dimBlock3(ColsPerBlk, 1, 1);
  dim3 dimGrid3((D + ColsPerBlk - 1) / ColsPerBlk, 1, 1);
  stddev_kernel<<<dimGrid3, dimBlock3>>>(std, std, D, N, sample);

  dim3 dimBlock4(1, 1, 1);
  dim3 dimGrid4(1, 1, 1);
  cudaMemsetAsync(std, 0, D * sizeof(Type));
  for (int i = 0; i < (D + ColsPerBlk - 1) / ColsPerBlk; i++) {
    stddev_kernel<<<dimBlock4, dimBlock4>>>(std, std, D, N, sample);
    cudaDeviceSynchronize();
  }

  for (int i = 0; i < D; i++) std[i] = sqrtf(std[i] / sampleSize);
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
  cudaMalloc((void**)&data, inputSizeByte);

  srand(123);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < D; j++)
      data[i * D + j] = rand() / (float)RAND_MAX;

  long outputSize = D;
  long outputSizeByte = outputSize * sizeof(float);
  float *std;
  cudaMalloc((void**)&std, outputSizeByte);
  float *std_ref;
  cudaMalloc((void**)&std_ref, outputSizeByte);

  cudaMemsetAsync(std, 0, outputSizeByte);
  cudaMemsetAsync(std_ref, 0, outputSizeByte);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    stddev(std, data, D, N, sample);
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of stddev kernels: %f (s)\n", (time * 1e-9f) / repeat);

  stddev(std_ref, data, D, N, sample);
  cudaDeviceSynchronize();

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