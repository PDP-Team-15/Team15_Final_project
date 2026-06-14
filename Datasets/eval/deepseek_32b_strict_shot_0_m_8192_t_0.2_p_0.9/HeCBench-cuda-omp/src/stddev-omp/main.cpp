#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "reference.h"



template <typename Type, typename IdxType>
void sampleKernel_omp(Type *std, IdxType D, IdxType N) {
  #pragma omp parallel for
  for (IdxType i = 0; i < D; ++i) {
    std[i] = sqrtf(std[i] / N);
  }
}



template <typename Type, typename IdxType, int TPB, int ColsPerBlk = 32>
void sopKernel_omp(
        Type *__restrict__ std, 
  const Type *__restrict__ data, 
  IdxType D, 
  IdxType N) 
{
  const int RowsPerThread = 4;
  const int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;
  
  #pragma omp parallel
  {
    Type thread_data[ColsPerBlk];
    for (int c = 0; c < ColsPerBlk; ++c) {
      thread_data[c] = Type(0);
    }
    
    IdxType colId = omp_get_thread_num() % ColsPerBlk;
    IdxType rowId = (omp_get_thread_num() / ColsPerBlk) + 
                    (omp_get_team_size() * omp_get_num_teams() / ColsPerBlk) * 
                    (omp_get_team_num() / (TPB / ColsPerBlk)));
    
    const IdxType stride = RowsPerBlk * omp_get_num_teams();
    
    for (IdxType i = rowId; i < N; i += stride) {
      if (colId < D) {
        Type val = data[i * D + colId];
        thread_data[colId % ColsPerBlk] += val * val;
      }
    }
    
    #pragma omp barrier
    
    if (colId < ColsPerBlk) {
      #pragma omp atomic
      std[colId] += thread_data[colId % ColsPerBlk];
    }
  }
}



template <typename Type, typename IdxType = int>
void stddev_omp(Type *std, const Type *data, IdxType D, IdxType N, bool sample) {
  static const int TPB = 256;
  static const int ColsPerBlk = 32;
  
  for (IdxType i = 0; i < D; ++i) {
    std[i] = 0;
  }
  
  sopKernel_omp<Type, IdxType, TPB, ColsPerBlk>(std, data, D, N);
  
  IdxType sampleSize = sample ? N-1 : N;
  sampleKernel_omp<Type, IdxType>(std, D, sampleSize);
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

  for (int i = 0; i < repeat; i++)
    stddev_omp(std, data, D, N, sample);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    stddev_omp(std, data, D, N, sample);
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