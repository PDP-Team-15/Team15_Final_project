```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "reference.h"

template <typename Type, typename IdxType>
void sampleKernel (Type *std, IdxType D, IdxType N) {
  IdxType i = omp_get_thread_num();
  if (i < D) std[i] = sqrtf(std[i] / N);
}

template <typename Type, typename IdxType, int TPB, int ColsPerBlk = 32>
void sopKernel(
        Type *__restrict__ std, 
  const Type *__restrict__ data, 
  IdxType D, 
  IdxType N) 
{
  const int RowsPerBlkPerIter = TPB / ColsPerBlk;
  IdxType thisColId = omp_get_thread_num() % ColsPerBlk;
  IdxType thisRowId = omp_get_thread_num() / ColsPerBlk;
  IdxType colId = thisColId + ((IdxType)omp_get_num_threads() / ColsPerBlk * ColsPerBlk);
  IdxType rowId = thisRowId + ((IdxType)omp_get_num_threads() / RowsPerBlkPerIter * RowsPerBlkPerIter);
  Type thread_data = Type(0);
  const IdxType stride = RowsPerBlkPerIter * omp_get_num_threads();
  for (IdxType i = rowId; i < N; i += stride) {
    Type val = (colId < D) ? data[i * D + colId] : Type(0);
    thread_data += val * val;
  }
  __attribute__((aligned(ColsPerBlk))) Type sstd[ColsPerBlk];
  if (omp_get_thread_num() < ColsPerBlk) sstd[threadIdx.x] = Type(0);
  #pragma omp barrier

  #pragma omp atomic
  sstd[thisColId] += thread_data;
  #pragma omp barrier

  if (omp_get_thread_num() < ColsPerBlk) {
    #pragma omp atomic
    std[colId] += sstd[thisColId];
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
    IdxType i = omp_get_thread_num();
    if (i < D) std[i] = 0;
  }

  #pragma omp parallel for num_threads(TPB)
  for (IdxType i = 0; i < D; i++) {
    Type thread_data = 0;
    for (IdxType j = i; j < N; j += RowsPerBlk) {
      Type val = (i < D) ? data[j * D + i] : Type(0);
      thread_data += val * val;
    }
    #pragma omp critical
    std[i] += thread_data;
  }

  IdxType sampleSize = sample ? N-1 : N;
  #pragma omp parallel for num_threads(TPB)
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

  #pragma omp parallel for
  for (int i = 0; i < D; i++) {
    std[i] = 0;
  }

  for (int i = 0; i < repeat; i++)
    stddev(std, data, D, N, sample);

  float *d_data;
  float *d_std;
  d_data = (float*) malloc (inputSizeByte);
  d_std = (float*) malloc (D * sizeof(float));

  srand(123);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < D; j++) 
      d_data[i*D + j] = rand() / (float)RAND_MAX; 

  stddev(d_std, d_data, D, N, sample);

  bool ok = true;
  for (int i = 0; i < D; i++) {
    if (fabsf(std[i] - d_std[i]) > 1e-3) {
      ok = false;
      break;
    }
  }

  stddev_ref(std_ref, data, D, N, sample);

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
  free(d_data);
  return 0;
}
```

Note that the OpenMP version of the code uses `omp_get_thread_num()` to get the thread ID, and `omp_get_num_threads()` to get the number of threads. The `#pragma omp barrier` directive is used to synchronize threads, and `#pragma omp atomic` is used to perform atomic operations. The `#pragma omp parallel for` directive is used to parallelize loops. The `#pragma omp critical` directive is used to synchronize threads in a critical section.