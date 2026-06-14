```c
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
  #pragma omp parallel for
  for (IdxType i = 0; i < N; i += TPB) {
    IdxType rowId = i / TPB;
    IdxType colId = omp_get_thread_num() % ColsPerBlk;
    Type thread_data = 0;
    const IdxType stride = TPB;
    for (IdxType j = rowId; j < N && j < rowId + stride; j++) {
      IdxType idx = j * D + colId;
      if (idx < D) {
        thread_data += data[idx] * data[idx];
      }
    }
    #pragma omp critical
    {
      if (colId < ColsPerBlk) {
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
  #pragma omp parallel for
  for (IdxType i = 0; i < D; i++) {
    std[i] = 0;
  }
  #pragma omp parallel for
  for (IdxType i = 0; i < N; i += TPB) {
    IdxType rowId = i / TPB;
    IdxType colId = omp_get_thread_num() % ColsPerBlk;
    Type thread_data = 0;
    const IdxType stride = TPB;
    for (IdxType j = rowId; j < N && j < rowId + stride; j++) {
      IdxType idx = j * D + colId;
      if (idx < D) {
        thread_data += data[idx] * data[idx];
      }
    }
    #pragma omp critical
    {
      if (colId < ColsPerBlk) {
        std[colId] += thread_data;
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
  float *data = (float*) malloc (D * N * sizeof(float));

  srand(123);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < D; j++) 
      data[i*D + j] = rand() / (float)RAND_MAX; 

  float *std  = (float*) malloc (D * sizeof(float));
  float *std_ref  = (float*) malloc (D * sizeof(float));

  #pragma omp parallel for
  for (int i = 0; i < D; i++) {
    std[i] = 0;
  }

  stddev(std, data, D, N, sample);

  for (int i = 0; i < repeat; i++)
    stddev(std, data, D, N, sample);

  float *std_ref_copy = (float*) malloc (D * sizeof(float));
  stddev_ref(std_ref_copy, data, D, N, sample);

  bool ok = true;
  for (int i = 0; i < D; i++) {
    if (fabsf(std_ref_copy[i] - std[i]) > 1e-3) {
      ok = false;
      break;
    }
  }

  printf("%s\n", ok ? "PASS" : "FAIL");
  free(std_ref_copy);
  free(std);
  free(std_ref);
  free(data);
  return 0;
}
```