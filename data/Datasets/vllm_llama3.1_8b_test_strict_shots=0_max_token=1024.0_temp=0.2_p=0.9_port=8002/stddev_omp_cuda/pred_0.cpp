```cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>

template <typename Type, typename IdxType = int>
__global__ void stddev_kernel(Type *std, const Type *data, IdxType D, IdxType N, bool sample) {
  __shared__ Type sstd[256];
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int gridDim_x = gridDim.x;

  const int RowsPerBlkPerIter = blockDim.x / 32;
  IdxType thisColId = tx % 32;
  IdxType thisRowId = tx / 32;
  IdxType colId = thisColId + ((IdxType)by * 32);
  IdxType rowId = thisRowId + ((IdxType)bx * RowsPerBlkPerIter);
  Type thread_data = Type(0);
  const IdxType stride = RowsPerBlkPerIter * gridDim_x;
  for (IdxType i = rowId; i < N; i += stride) {
    Type val = (colId < D) ? data[i * D + colId] : Type(0);
    thread_data += val * val;
  }
  if (tx < 32) sstd[threadIdx.x] = Type(0);
  __syncthreads();

  __shared__ __cuda_atomic Type *sstd_ptr;
  sstd_ptr = sstd;
  if (threadIdx.x < 32) {
    atomicAdd(sstd_ptr + threadIdx.x, thread_data);
  }
  __syncthreads();

  if (tx < 32) {
    atomicAdd(std + colId, sstd[threadIdx.x]);
  }
}

template <typename Type, typename IdxType = int>
void stddev(Type *std, const Type *data, IdxType D, IdxType N, bool sample) {
  int TPB = 256;
  int RowsPerThread = 4;
  int ColsPerBlk = 32;
  int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;

  int TeamX = (N + (IdxType)RowsPerBlk - 1) / (IdxType)RowsPerBlk;
  int TeamY = (D + (IdxType)ColsPerBlk - 1) / (IdxType)ColsPerBlk;
  int Teams = TeamX * TeamY;

  cudaMemset(std, 0, D * sizeof(Type));

  dim3 grid(TeamX, TeamY);
  dim3 block(TPB, 1);
  stddev_kernel<<<grid, block>>>(std, data, D, N, sample);

  IdxType sampleSize = sample ? N-1 : N;
  cudaMemset(std, 0, D * sizeof(Type));
  grid.x = Teams;
  block.x = TPB;
  stddev_kernel<<<grid, block>>>(std, data, D, N, sample);

  for (int i = 0; i < D; i++)
    std[i] = sqrtf(std[i] / sampleSize);
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

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  for (int i = 0; i < repeat; i++)
    stddev(std, data, D, N, sample);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
 