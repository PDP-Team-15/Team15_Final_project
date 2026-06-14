You are an HPC expert specializing in translating between parallel programming APIs. Translate precisely and completely, preserving all function qualifiers (e.g., const, inline), template parameters, and macro definitions. Output only the translated code inside a code block. No explanations, no comments, no omissions.

When translating to CUDA:
1. Always use proper __host__, __device__, or __global__ qualifiers based on function scope
2. Preserve const qualifiers in function parameters
3. Ensure all #define directives are correctly translated
4. Validate CUDA-specific constructs like memory operations, kernel launches, and stream usage
5. Default to compile-time safety when converting run-time parameters
For each kernel code provided, translate it from cuda to omp. Ensure the translated code is complete, correct, and compiles without errors. Follow these guidelines:

1. Preserve all original functionality and algorithmic logic
2. Replace cuda with omp functions and constructs
3. Correctly declare and initialize all variables
4. Use proper CUDA syntax for kernel launches and memory management
5. Ensure all code is properly terminated with semicolons
6. Include necessary headers
7. Validate code structure and syntax before providing output
Translate the following code  from cuda to omp:
main.cu:



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>
#include "reference.h"



template <typename Type, typename IdxType>
__global__ void sampleKernel (Type *std, IdxType D, IdxType N) {
  IdxType i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < D) std[i] = sqrtf(std[i] / N);
}



template <typename Type, typename IdxType, int TPB, int ColsPerBlk = 32>
__global__ void sopKernel(
        Type *__restrict__ std, 
  const Type *__restrict__ data, 
  IdxType D, 
  IdxType N) 
{
  const int RowsPerBlkPerIter = TPB / ColsPerBlk;
  IdxType thisColId = threadIdx.x % ColsPerBlk;
  IdxType thisRowId = threadIdx.x / ColsPerBlk;
  IdxType colId = thisColId + ((IdxType)blockIdx.y * ColsPerBlk);
  IdxType rowId = thisRowId + ((IdxType)blockIdx.x * RowsPerBlkPerIter);
  Type thread_data = Type(0);
  const IdxType stride = RowsPerBlkPerIter * gridDim.x;
  for (IdxType i = rowId; i < N; i += stride) {
    Type val = (colId < D) ? data[i * D + colId] : Type(0);
    thread_data += val * val;
  }
  __shared__ Type sstd[ColsPerBlk];
  if (threadIdx.x < ColsPerBlk) sstd[threadIdx.x] = Type(0);
  __syncthreads();

  atomicAdd(sstd + thisColId, thread_data);
  __syncthreads();

  if (threadIdx.x < ColsPerBlk) atomicAdd(std + colId, sstd[thisColId]);
}



template <typename Type, typename IdxType = int>
void stddev(Type *std, const Type *data, IdxType D, IdxType N, bool sample) {
  static const int TPB = 256;
  static const int RowsPerThread = 4;
  static const int ColsPerBlk = 32;
  static const int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;
  dim3 grid((N + (IdxType)RowsPerBlk - 1) / (IdxType)RowsPerBlk, 
            (D + (IdxType)ColsPerBlk - 1) / (IdxType)ColsPerBlk);
  dim3 block(TPB);

  cudaMemset(std, 0, sizeof(Type) * D); 


  sopKernel<Type, IdxType, TPB, ColsPerBlk> <<<grid, block>>>(std, data, D, N);

  IdxType sampleSize = sample ? N-1 : N;
  sampleKernel<Type, IdxType> <<<(D+TPB-1)/TPB, TPB>>>(std, D, sampleSize);
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

  float *d_data;
  cudaMalloc((void**)&d_data, inputSizeByte);
  cudaMemcpy(d_data, data, inputSizeByte, cudaMemcpyHostToDevice);

  

  long outputSize = D;
  long outputSizeByte = outputSize * sizeof(float);
  float *std  = (float*) malloc (outputSizeByte);
  float *std_ref  = (float*) malloc (outputSizeByte);
  float *d_std;
  cudaMalloc((void**)&d_std, outputSizeByte);

  

  stddev(d_std, d_data, D, N, sample);

  

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    stddev(d_std, d_data, D, N, sample);

  cudaDeviceSynchronize();
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
