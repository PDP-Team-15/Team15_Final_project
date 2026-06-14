#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "reference.h"

template <typename Type, typename IdxType>
void sampleKernel (Type *std, IdxType D, IdxType N) {
  #pragma omp parallel for
  for (IdxType i = 0; i < D; i++) {
    std[i] = sqrtf(std[i] / N);
  }
}

template <typename Type, typename IdxType, int TPB, int ColsPerBlk = 32>
void sopKernel(
        Type *__restrict__ std, 
  const Type *__restrict__ data, 
  IdxType D, 
  IdxType N) 
{
  #pragma omp parallel for num_threads(TPB)
  for (IdxType i = 0; i < N; i += TPB) {
    IdxType rowId = i;
    for (IdxType j = 0; j < TPB && rowId < N; j++) {
      Type thread_data = 0;
      for (IdxType k = 0; k < D; k += ColsPerBlk) {
        IdxType colId = k + (IdxType)omp_get_thread_num() % ColsPerBlk;
        if (colId < D) {
          thread_data += data[rowId * D + colId] * data[rowId * D + colId];
        }
      }
      #pragma omp critical
      {
        if (omp_get_thread_num() < ColsPerBlk) {
          std[colId + (IdxType)omp_get_thread_num()] += thread_data;
        }
      }
    }
  }
  #pragma omp parallel for
  for (IdxType i = 0; i < D; i++) {
    if (i < ColsPerBlk) {
      std[i] /= ColsPerBlk;
    }
  }
}

template <typename Type, typename IdxType = int>
void stddev(Type *std, const Type *data, IdxType D, IdxType N, bool sample) {
  static const int TPB = 256;
  static const int RowsPerThread = 4;
  static const int ColsPerBlk = 32;
  static const int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;
  #pragma omp parallel num_threads((N + RowsPerBlk - 1) / RowsPerBlk)
  {
    IdxType tid = omp_get_thread_num();
    IdxType rowId = tid * RowsPerBlk;
    for (; rowId < N; rowId += RowsPerBlk) {
      for (IdxType i = 0; i < D; i++) {
        std[i] = 0;
      }
      sopKernel<Type, IdxType, TPB, ColsPerBlk>(std, data, D, N);
      if (sample) {
        sampleKernel<Type, IdxType>(std, D, N-1);
      } else {
        sampleKernel<Type, IdxType>(std, D, N);
      }
    }
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