#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include "reference.h"

template <typename Type, typename IdxType>
void sampleKernelOMP(Type *std, IdxType D, IdxType N) {
    #pragma omp parallel for
    for (IdxType i = 0; i < D; ++i) {
        std[i] = sqrtf(std[i] / N);
    }
}

template <typename Type, typename IdxType>
void sopKernelOMP(Type *std, const Type *data, IdxType D, IdxType N) {
    #pragma omp parallel for reduction(+:std[:D])
    for (IdxType col = 0; col < D; ++col) {
        Type sum = 0.0f;
        for (IdxType row = 0; row < N; ++row) {
            IdxType index = row * D + col;
            sum += data[index] * data[index];
        }
        std[col] = sum;
    }
}

template <typename Type, typename IdxType = int>
void stddev(Type *std, const Type *data, IdxType D, IdxType N, bool sample) {
    std::fill(std, std + D, Type(0));
    sopKernelOMP(std, data, D, N);
    IdxType sampleSize = sample ? N - 1 : N;
    sampleKernelOMP(std, D, sampleSize);
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <D> <N> <repeat>\n", argv[0]);
    printf("D: number of columns of data (must be a multiple of 32)\n");
    printf("N: number of rows of data (at least one row)\n");
    return 1;
  }
  int D = atoi(argv[1]); // columns must be a multiple of 32
  int N = atoi(argv[2]); // at least one row
  int repeat = atoi(argv[3]);

  bool sample = true;
  long inputSize = D * N;
  long inputSizeByte = inputSize * sizeof(float);
  float *data = (float*) malloc (inputSizeByte);

  // input data 
  srand(123);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < D; j++) 
      data[i*D + j] = rand() / (float)RAND_MAX; 

  // host and device results
  long outputSize = D;
  long outputSizeByte = outputSize * sizeof(float);
  float *std  = (float*) malloc (outputSizeByte);
  float *std_ref  = (float*) malloc (outputSizeByte);

  #pragma omp target data map (to: data[0:inputSize]) map (from: std[0:outputSize])
  {
    // warmup
    stddev(std, data, D, N, sample);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      stddev(std, data, D, N, sample);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of stddev kernels: %f (s)\n", (time * 1e-9f) / repeat);
  }

  // verify
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