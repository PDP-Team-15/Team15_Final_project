#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "reference.h"

template <typename Type, typename IdxType>
void sampleKernel(Type *std, IdxType D, IdxType N) {
  #pragma omp parallel for
  for (IdxType i = 0; i < D; ++i) {
    std[i] = sqrtf(std[i] / N);
  }
}

template <typename Type, typename IdxType, int TPB, int ColsPerBlk = 32>
void sopKernel(Type *std, const Type *data, IdxType D, IdxType N) {
  const int RowsPerBlkPerIter = TPB / ColsPerBlk;
  const int RowsPerThread = 4;
  const int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;
  
  #pragma omp parallel
  {
    Type thread_data = 0;
    int thread_id = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    
    int thisColId = thread_id % ColsPerBlk;
    int thisRowId = thread_id / ColsPerBlk;
    
    int colId = thisColId + ((IdxType)(thread_id / (TPB / ColsPerBlk)) * ColsPerBlk);
    int rowId = thisRowId + ((IdxType)(thread_id / (TPB / ColsPerBlk)) * RowsPerBlkPerIter);
    
    const IdxType stride = RowsPerBlkPerIter * num_threads;
    
    #pragma omp for
    for (IdxType i = rowId; i < N; i += stride) {
      if (colId < D) {
        Type val = data[i * D + colId];
        thread_data += val * val;
      }
    }
    
    #pragma omp critical
    {
      if (thread_id < ColsPerBlk) {
        std[colId] += thread_data;
      }
    }
  }
}

template <typename Type, typename IdxType = int>
void stddev(Type *std, const Type *data, IdxType D, IdxType N, bool sample) {
  const int TPB = 256;
  const int ColsPerBlk = 32;
  
  #pragma omp parallel for
  for (IdxType i = 0; i < D; ++i) {
    std[i] = 0;
  }
  
  sopKernel<Type, IdxType, TPB, ColsPerBlk>(std, data, D, N);
  
  IdxType sampleSize = sample ? N - 1 : N;
  sampleKernel<Type, IdxType>(std, D, sampleSize);
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
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < D; ++j) {
      data[i * D + j] = rand() / (float)RAND_MAX;
    }
  }
  
  long outputSize = D;
  long outputSizeByte = outputSize * sizeof(float);
  float *std = (float*)malloc(outputSizeByte);
  float *std_ref = (float*)malloc(outputSizeByte);
  
  #pragma omp parallel for
  for (int i = 0; i < D; ++i) {
    std[i] = 0;
  }
  
  auto start = std::chrono::steady_clock::now();
  
  for (int i = 0; i < repeat; ++i) {
    stddev(std, data, D, N, sample);
  }
  
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of stddev kernels: %f (s)\n", (time * 1e-9f) / repeat);
  
  stddev_ref(std_ref, data, D, N, sample);
  
  bool ok = true;
  for (int i = 0; i < D; ++i) {
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