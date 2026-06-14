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
void sopKernel(
        Type *__restrict__ std, 
  const Type *__restrict__ data, 
  IdxType D, 
  IdxType N) 
{
  #pragma omp parallel for
  for (IdxType i = 0; i < N; i += TPB) {
    IdxType rowId = i;
    for (IdxType j = 0; j < TPB && rowId < N; j++) {
      IdxType colId = j % ColsPerBlk;
      IdxType thisColId = colId;
      IdxType thisRowId = j / ColsPerBlk;
      Type thread_data = Type(0);
      for (IdxType k = 0; k < ColsPerBlk && rowId + k * TPB / ColsPerBlk < N; k++) {
        IdxType stride = TPB / ColsPerBlk * omp_get_num_threads();
        for (IdxType l = 0; l < TPB / ColsPerBlk && rowId + k * TPB / ColsPerBlk + l * stride < N; l++) {
          IdxType row = rowId + k * TPB / ColsPerBlk + l * stride;
          Type val = (colId < D) ? data[row * D + colId] : Type(0);
          thread_data += val * val;
        }
      }
      #pragma omp critical
      {
        if (thisColId < ColsPerBlk) {
          std[colId] += thread_data;
        }
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
  #pragma omp parallel
  {
    #pragma omp for
    for (IdxType i = 0; i < D; i++) {
      std[i] = 0;
    }
    #pragma omp for
    for (IdxType i = 0; i < N; i++) {
      for (IdxType j = 0; j < D; j++) {
        std[j] += data[i * D + j] * data[i * D + j];
      }
    }
    #pragma omp for
    for (IdxType i = 0; i < D; i++) {
      std[i] /= N;
    }
    if (sample) {
      #pragma omp for
      for (IdxType i = 0; i < D; i++) {
        std[i] = sqrtf(std[i] / (N - 1));
      }
    } else {
      #pragma omp for
      for (IdxType i = 0; i < D; i++) {
        std[i] = sqrtf(std[i] / N);
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

  #pragma omp parallel for
  for (int i = 0; i < D; i++) {
    std[i] = 0;
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < D; j++) {
      std[j] += data[i * D + j] * data[i * D + j];
    }
  }

  for (int i = 0; i < D; i++) {
    std[i] /= N;
  }

  if (sample) {
    for (int i = 0; i < D; i++) {
      std[i] = sqrtf(std[i] / (N - 1));
    }
  } else {
    for (int i = 0; i < D; i++) {
      std[i] = sqrtf(std[i] / N);
    }
  }

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
```