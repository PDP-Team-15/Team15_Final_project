#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <chrono>
#include "reference.h"

template <typename Type, typename IdxType>
void sampleKernel(Type *std, IdxType D, IdxType N) {
  #pragma omp parallel for reduction(+:std[:D])
  for (IdxType i = 0; i < D; ++i) {
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
  const int RowsPerBlkPerIter = TPB / ColsPerBlk;
  IdxType thisColId = omp_get_thread_num() % ColsPerBlk;
  IdxType thisRowId = omp_get_thread_num() / ColsPerBlk;
  IdxType colId = thisColId + ((IdxType)omp_get_team_num() * ColsPerBlk);
  IdxType rowId = thisRowId + ((IdxType)omp_get_thread_num() * RowsPerBlkPerIter);
  Type thread_data = Type(0);
  const IdxType stride = RowsPerBlkPerIter * omp_get_num_teams();
  for (IdxType i = rowId; i < N; i += stride) {
    Type val = (colId < D) ? data[i * D + colId] : Type(0);
    thread_data += val * val;
  }
  #pragma omp barrier

  #pragma omp team shared(sstd)
  {
    Type sstd[ColsPerBlk];
    if (thisColId < ColsPerBlk) sstd[thisColId] = Type(0);
    #pragma omp distribute parallel for reduction(+:sstd[:ColsPerBlk])
    for (IdxType i = 0; i < ColsPerBlk; ++i) {
      sstd[i] += thread_data;
    }
    #pragma omp single
    {
      for (IdxType i = 0; i < ColsPerBlk; ++i) {
        atomicAdd(std + colId, sstd[i]);
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
  #pragma omp target teams num_teams((N + (IdxType)RowsPerBlk - 1) / (IdxType)RowsPerBlk) thread_limit(TPB)
  {
    #pragma omp distribute parallel for
    for (IdxType i = 0; i < D; ++i) {
      std[i] = Type(0);
    }
    sopKernel<Type, IdxType, TPB, ColsPerBlk> (std, data, D, N);
  }

  IdxType sampleSize = sample ? N-1 : N;
  sampleKernel<Type, IdxType> (std, D, sampleSize);
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
  float *std  = (float*) malloc (outputSizeByte);
  float *std_ref  = (float*) malloc (outputSizeByte);
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