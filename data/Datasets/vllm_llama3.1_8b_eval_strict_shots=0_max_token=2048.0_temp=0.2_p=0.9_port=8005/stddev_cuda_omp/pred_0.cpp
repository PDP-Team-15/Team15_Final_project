main.cpp:



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "reference.h"



template <typename Type, typename IdxType>
void sampleKernel (Type *std, IdxType D, IdxType N) {
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (IdxType i = 0; i < D; i++) std[i] = sqrtf(std[i] / N);
}



template <typename Type, typename IdxType, int TPB, int ColsPerBlk = 32>
void sopKernel(
        Type *__restrict__ std, 
  const Type *__restrict__ data, 
  IdxType D, 
  IdxType N) 
{
  const int RowsPerBlkPerIter = TPB / ColsPerBlk;
  IdxType thisColId = threadIdx.x % ColsPerBlk;
  IdxType thisRowId = threadIdx.x / ColsPerBlk;
  IdxType colId = thisColId + ((IdxType)omp_get_team_num()) * ColsPerBlk;
  IdxType rowId = thisRowId + ((IdxType)omp_get_num_teams()) * RowsPerBlkPerIter;
  Type thread_data = Type(0);
  const IdxType stride = RowsPerBlkPerIter * omp_get_num_teams();
  for (IdxType i = rowId; i < N; i += stride) {
    Type val = (colId < D) ? data[i * D + colId] : Type(0);
    thread_data += val * val;
  }
  #pragma omp target 
  Type sstd[ColsPerBlk];
  #pragma omp parallel 
  {
    #pragma omp barrier
    sstd[thisColId] = Type(0);
    #pragma omp barrier

    #pragma omp atomic update
    sstd[thisColId] += thread_data;
    #pragma omp barrier

    if (thisColId == 0) {
      #pragma omp atomic update
      std[colId] += sstd[0];
    }
  }
}



template <typename Type, typename IdxType = int>
void stddev(Type *std, const Type *data, IdxType D, IdxType N, bool sample) {
  static const int TPB = 256;
  static const int RowsPerThread = 4;
  static const int ColsPerBlk = 32;
  static const int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;
  const IdxType numTeams = (N + RowsPerBlk - 1) / RowsPerBlk;
  const IdxType numBlocks = (D + ColsPerBlk - 1) / ColsPerBlk;

  #pragma omp target map(alloc: std[0:D])
  for (int i = 0; i < D; i++) std[i] = 0.f; 


  sopKernel<Type, IdxType, TPB, ColsPerBlk>(std, data, D, N, numTeams, numBlocks);

  IdxType sampleSize = sample ? N-1 : N;
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
  float *data = (float*) malloc (inputSizeByte);

  

  srand(123);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < D; j++) 
      data[i*D + j] = rand() / (float)RAND_MAX; 

  float *std  = (float*) malloc (D * sizeof(float));
  float *std_ref  = (float*) malloc (D * sizeof(float));

  

  stddev(std, data, D, N, sample);

  

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    stddev(std, data, D, N, sample);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of stddev kernels: %f (s)\n", (time * 1e-9f) / repeat);

  

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