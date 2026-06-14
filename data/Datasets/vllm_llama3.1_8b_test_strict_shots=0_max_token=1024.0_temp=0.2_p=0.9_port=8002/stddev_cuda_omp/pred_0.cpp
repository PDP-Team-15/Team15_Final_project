```cpp
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
    const int RowsPerBlkPerIter = TPB / ColsPerBlk;
    const int threadId = omp_get_thread_num();
    const int thisColId = threadId % ColsPerBlk;
    const int thisRowId = threadId / ColsPerBlk;
    const int colId = thisColId + ((IdxType)omp_get_num_threads() / ColsPerBlk * ColsPerBlk);
    const int rowId = thisRowId + ((IdxType)omp_get_num_threads() / RowsPerBlkPerIter * RowsPerBlkPerIter);
    Type thread_data = Type(0);
    const IdxType stride = RowsPerBlkPerIter * omp_get_num_threads();
    for (IdxType j = rowId; j < N && j < i + TPB; j += stride) {
      Type val = (colId < D) ? data[j * D + colId] : Type(0);
      thread_data += val * val;
    }
    #pragma omp critical
    {
      std[colId] += thread_data;
    }
  }
}

template <typename Type, typename IdxType = int>
void stddev(Type *std, const Type *data, IdxType D, IdxType N, bool sample) {
  static const int TPB = 256;
  static const int RowsPerThread = 4;
  static const int ColsPerBlk = 32;
  static const int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;
  #pragma omp parallel num_threads(TPB)
  {
    #pragma omp master
    {
      cudaMemset(std, 0, sizeof(Type) * D); 
    }
    #pragma omp barrier
    sopKernel<Type, IdxType, TPB, ColsPerBlk>(std, data, D, N);
    IdxType sampleSize = sample ? N-1 : N;
    sampleKernel<Type, IdxType>(std, D, sampleSize);
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

  

  float *data_ref = (float*) malloc (inputSizeByte);
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