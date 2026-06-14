#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <chrono>
#include <omp.h>

#define VECTOR_SIZE (8*1024*1024)
#define granularity (8)
#define fusion_degree (4)
#define seed 0.1f

void benchmark_func(float *g_data, const int compute_iterations) {
  const int blockSize = 256;
  const int stride = blockSize;
  const int big_stride = 128 * blockSize * granularity;

  #pragma omp parallel for collapse(2)
  for(int k = 0; k < fusion_degree; k++) {
    for(int idx = 0; idx < VECTOR_SIZE / granularity; idx++) {
      float tmps[granularity];
      for(int j = 0; j < granularity; j++) {
        tmps[j] = g_data[idx * granularity + j + k * big_stride];
        for(int i = 0; i < compute_iterations; i++)
          tmps[j] = tmps[j] * tmps[j] + seed;
      }

      float sum = 0.f;
      for(int j = 0; j < granularity; j += 2)
        sum += tmps[j] * tmps[j + 1];

      for(int j = 0; j < granularity; j++)
        g_data[idx * granularity + j + k * big_stride] = sum;
    }
  }
}

void mixbenchGPU(long size, int repeat) {
  const char *benchtype = "compute with global memory (block strided)";
  printf("Trade-off type:%s\n", benchtype);
  float *cd = (float*) malloc(size * sizeof(float));
  for (int i = 0; i < size; i++) cd[i] = 0;

  const long reduced_grid_size = size / granularity / 128;
  const int block_dim = 256;
  const int grid_dim = reduced_grid_size / block_dim;

  float *d_cd = (float*) malloc(size * sizeof(float));
  memcpy(d_cd, cd, size * sizeof(float));

  for (int i = 0; i < repeat; i++) {
    benchmark_func(d_cd, i);
  }

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    benchmark_func(d_cd, i);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time: %f (s)\n", time * 1e-9f);

  memcpy(cd, d_cd, size * sizeof(float));

  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (cd[i] != 0) {
      if (fabsf(cd[i] - 0.050807f) > 1e-6f) {
        ok = false;
        printf("Verification failed at index %d: %f\n", i, cd[i]);
        break;
      }
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(cd);
  free(d_cd);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  unsigned int datasize = VECTOR_SIZE * sizeof(float);

  printf("Buffer size: %dMB\n", datasize / (1024 * 1024));

  mixbenchGPU(VECTOR_SIZE, repeat);

  return 0;
}