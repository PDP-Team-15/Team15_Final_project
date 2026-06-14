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

template <typename Type, typename IdxType = int>
__global__ void stddev_kernel(Type *std, const Type *data, IdxType D, IdxType N, bool sample) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int bx = blockIdx.y;
  int by = blockIdx.z;
  int gridDim_x = gridDim.y;

  const int RowsPerBlkPerIter = blockDim.x / 32;
  IdxType thisColId = tx % 32;
  IdxType thisRowId = tx / 32;
  IdxType colId = thisColId + ((IdxType)by * 32);
  IdxType rowId = thisRowId + ((IdxType)bx * RowsPerBlkPerIter);
  Type thread_data = Type(0);
  const IdxType stride = RowsPerBlkPerIter * gridDim.x;
  for (IdxType i = rowId; i < N; i += stride) {
    Type val = (colId < D) ? data[i * D + colId] : Type(0);
    thread_data += val * val;
  }
  if (tx < 32) atomicAdd(&std[colId], thread_data);
}

template <typename Type, typename IdxType = int>
void stddev(Type *std, const Type *data, IdxType D, IdxType N, bool sample) {
  int blockSize = 256;
  int numBlocks = (D + 31) / 32;
  int numGrids = (N + 31) / 256;

  // Initialize the result array
  for (int i = 0; i < D; i++) std[i] = Type(0);

  // Calculate the sum of squares for each column
  dim3 dimGrid(numGrids, numBlocks, 1);
  dim3 dimBlock(blockSize, 1, 1);
  stddev_kernel<<<dimGrid, dimBlock>>>(std, data, D, N, sample);

  // Calculate the standard deviation
  IdxType sampleSize = sample ? N-1 : N;
  for (int i = 0; i < D; i++) {
    std[i] = sqrtf(std[i] / sampleSize);
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
  float *data = (float*) malloc (inputSizeByte);

  srand(123);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < D; j++) 
      data[i*D + j] = rand() / (float)RAND_MAX; 

  long outputSize = D;
  long outputSizeByte = outputSize * sizeof(float);
  float *std  = (float*) malloc (outputSizeByte);
  float *std_ref  = (float*) malloc (outputSizeByte);

  // Run the CUDA kernel
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  for (int i = 0; i < repeat; i++)
    stddev(std, data, D, N, sample);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float time;
  cudaEventElapsedTime(&time, start, stop);
  printf("Average execution time of stddev kernels: %f (ms)\n", time / repeat);

  // Run the reference implementation
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
  return 0;
}