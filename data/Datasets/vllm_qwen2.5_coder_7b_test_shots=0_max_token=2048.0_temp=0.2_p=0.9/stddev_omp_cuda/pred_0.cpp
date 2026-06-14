Here is the CUDA translation of the provided OpenMP code:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

template <typename Type, typename IdxType = int>
__global__ void stddev_kernel(Type *std, const Type *data, IdxType D, IdxType N, bool sample) {
  __shared__ Type sstd[256];

  int tid = threadIdx.x;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int gridDim_x = gridDim.x;

  const int RowsPerBlkPerIter = 256 / 32;
  IdxType thisColId = tid % 32;
  IdxType thisRowId = tid / 32;
  IdxType colId = thisColId + ((IdxType)by * 32);
  IdxType rowId = thisRowId + ((IdxType)bx * RowsPerBlkPerIter);
  Type thread_data = Type(0);
  const IdxType stride = RowsPerBlkPerIter * gridDim_x;

  for (IdxType i = rowId; i < N; i += stride) {
    Type val = (colId < D) ? data[i * D + colId] : Type(0);
    thread_data += val * val;
  }

  sstd[tid] = thread_data;

  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sstd[tid] += sstd[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(&std[colId], sstd[0]);
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

  cudaMemset(std, 0, D * sizeof(Type));

  dim3 blockSize(ColsPerBlk, 1);
  dim3 gridSize(TeamX, TeamY);

  stddev_kernel<<<gridSize, blockSize>>>(std, data, D, N, sample);

  IdxType sampleSize = sample ? N-1 : N;
  cudaMemset(std, 0, D * sizeof(Type));

  for (int i = 0; i < repeat; i++) {
    stddev_kernel<<<gridSize, blockSize>>>(std, data, D, N, sample);
  }

  cudaDeviceSynchronize();

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
  float *d_data;
  cudaMalloc((void**)&d_data, inputSizeByte);

  srand(123);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < D; j++) 
      data[i*D + j] = rand() / (float)RAND_MAX; 

  cudaMemcpy(d_data, data, inputSizeByte, cudaMemcpyHostToDevice);

  long outputSize = D;
  long outputSizeByte = outputSize * sizeof(float);
  float *std  = (float*) malloc (outputSizeByte);
  float *d_std;
  cudaMalloc((void**)&d_std, outputSizeByte);

  cudaMemset(d_std, 0, outputSizeByte);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    stddev(d_std, d_data, D, N, sample);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of stddev kernels: %f (s)\n", (time * 1e-9f) / repeat);

  cudaMemcpy(std, d_std, outputSizeByte, cudaMemcpyDeviceToHost);

  float *std_ref  = (float*) malloc (outputSizeByte);
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
```

### Explanation:
1. **Kernel Function (`stddev_kernel`)**:
   - The kernel function is designed to compute the sum of squares for each column.
   - It uses shared memory to accumulate partial sums within each block and then reduces these sums across blocks using `__syncthreads()` and atomic operations.

2. **Main Function**:
   - Memory allocation and initialization are handled similarly to the original code.
   - Data is copied to the device before computation.
   - The kernel is launched with appropriate grid and block dimensions.
   - Results are copied back to the host after computation.
   - Timing and verification steps remain unchanged.

This CUDA version should provide similar performance and correctness as the original OpenMP code.