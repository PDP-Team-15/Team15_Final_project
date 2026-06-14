```cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "reference.h"

template <typename Type, typename IdxType>
void sampleKernel(Type *std, IdxType D, IdxType N) {
  #pragma omp parallel for
  for (IdxType i = 0; i < D; i++) {
    std[i] = sqrtf(std[i] / N);
  }
}

template <typename Type, typename IdxType, int TPB, int ColsPerBlk = 32>
void sopKernel(Type *__restrict__ std, 
               const Type *__restrict__ data, 
               IdxType D, 
               IdxType N) 
{
  #pragma omp parallel for num_threads(TPB)
  for (IdxType i = 0; i < N; i += TPB) {
    for (IdxType colId = 0; colId < D; colId++) {
      Type thread_data = 0;
      for (IdxType j = i; j < i + TPB && j < N; j++) {
        Type val = (colId < D) ? data[j * D + colId] : 0;
        thread_data += val * val;
      }
      #pragma omp critical
      {
        std[colId] += thread_data;
      }
    }
  }
}

template <typename Type, typename IdxType = int>
void stddev(Type *std, const Type *data, IdxType D, IdxType N, bool sample) {
  static const int TPB = 256;
  static const int RowsPerThread = 4;
  static const int ColsPerBlk = 32;
  static const int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;
  int grid = (N + RowsPerBlk - 1) / RowsPerBlk;
  int block = TPB;

  for (int i = 0; i < D; i++) {
    std[i] = 0;
  }

  #pragma omp parallel for num_threads(grid)
  for (int i = 0; i < grid; i++) {
    #pragma omp parallel for
    for (int j = 0; j < RowsPerBlk; j++) {
      int rowId = j + i * RowsPerBlk;
      for (int k = 0; k < TPB; k++) {
        int colId = k % ColsPerBlk;
        int thisRowId = k / ColsPerBlk;
        if (rowId < N) {
          Type val = (colId < D) ? data[rowId * D + colId] : 0;
          std[colId] += val * val;
        }
      }
    }
  }

  IdxType sampleSize = sample ? N-1 : N;
  #pragma omp parallel for
  for (IdxType i = 0; i < D; i++) {
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

  float *std  = (float*) malloc (D * sizeof(float));
  float *std_ref  = (float*) malloc (D * sizeof(float));

  stddev(std, data, D, N, sample);

  

  long outputSize = D;
  long outputSizeByte = outputSize * sizeof(float);
  float *d_data;
  float *d_std;
  cudaMalloc((void**)&d_data, inputSizeByte);
  cudaMalloc((void**)&d_std, outputSizeByte);
  cudaMemcpy(d_data, data, inputSizeByte, cudaMemcpyHostToDevice);

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
```