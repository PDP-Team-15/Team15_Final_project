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
  #pragma omp parallel for default(none) shared(std, D, N)
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
  #pragma omp parallel for default(none) shared(std, data, D, N) num_threads(TPB)
  for (IdxType rowId = 0; rowId < N; rowId += TPB) {
    for (IdxType colId = 0; colId < D; colId += ColsPerBlk) {
      Type thread_data = Type(0);
      for (IdxType i = rowId; i < N && i < rowId + TPB; i++) {
        Type val = (colId < D) ? data[i * D + colId] : Type(0);
        thread_data += val * val;
      }
      #pragma omp critical
      {
        if (colId < D) {
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
  IdxType grid_size = (N + RowsPerBlk - 1) / RowsPerBlk;
  IdxType block_size = TPB;

  #pragma omp parallel for default(none) shared(std, data, D, N)
  for (IdxType i = 0; i < D; i++) {
    std[i] = 0;
  }

  #pragma omp parallel for default(none) shared(std, data, D, N) num_threads(TPB)
  for (IdxType rowId = 0; rowId < N; rowId += TPB) {
    for (IdxType colId = 0; colId < D; colId += ColsPerBlk) {
      Type thread_data = Type(0);
      for (IdxType i = rowId; i < N && i < rowId + TPB; i++) {
        Type val = (colId < D) ? data[i * D + colId] : Type(0);
        thread_data += val * val;
      }
      #pragma omp critical
      {
        if (colId < D) {
          std[colId] += thread_data;
        }
      }
    }
  }

  IdxType sampleSize = sample ? N-1 : N;
  #pragma omp parallel for default(none) shared(std, D, sampleSize)
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

  #pragma omp parallel for default(none) shared(std, data, D, N)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < D; j++) {
      std[j] += data[i*D + j] * data[i*D + j];
    }
  }

  #pragma omp parallel for default(none) shared(std, D)
  for (int i = 0; i < D; i++) {
    std[i] = sqrtf(std[i] / N);
  }

  float *d_data;
  float *d_std;
  // Note: OpenMP does not have a direct equivalent to CUDA's cudaMemcpy
  // We will use a simple loop to copy the data
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < D; j++) {
      d_data[i*D + j] = data[i*D + j];
    }
  }

  for (int i = 0; i < D; i++) {
    d_std[i] = std[i];
  }

  #pragma omp parallel for default(none) shared(d_std, d_data, D, N)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < D; j++) {
      d_std[j] += d_data[i*D + j] * d_data[i*D + j];
    }
  }

  #pragma omp parallel for default(none) shared(d_std, D)
  for (int i = 0; i < D; i++) {
    d_std[i] = sqrtf(d_std[i] / N);
  }

  stddev(d_std, d_data, D, N, sample);

  #pragma omp parallel for default(none) shared(d_std, D, sample)
  for (int i = 0; i < D; i++) {
    d_std[i] = sqrtf(d_std[i] / (sample ? N-1 : N));
  }

  bool ok = true;
  for (int i = 0; i < D; i++) {
    if (fabsf(std_ref[i] - d_std[i]) > 1e-3) {
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