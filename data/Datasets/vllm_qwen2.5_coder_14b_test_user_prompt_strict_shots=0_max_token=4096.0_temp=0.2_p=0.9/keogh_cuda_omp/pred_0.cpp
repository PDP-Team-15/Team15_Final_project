```cpp
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "reference.h"

#define CACHE_SIZE 1024

void lb_keogh(const float *__restrict__ subject,
              const float *__restrict__ avgs,
              const float *__restrict__ stds, 
                    float *__restrict__ lb_keogh,
              const float *__restrict__ lower_bound,
              const float *__restrict__ upper_bound,
              const int M,
              const int N) 
{
  #pragma omp parallel
  {
    int lid = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    int blockSize = num_threads;
    int idx = blockSize * (lid / num_threads) + (lid % num_threads);

    float cache[CACHE_SIZE];

    for (int k = lid % num_threads; k < M + num_threads; k += num_threads)
      if (blockSize * (lid / num_threads) + k < N) cache[k] = subject[blockSize * (lid / num_threads) + k];

    #pragma omp barrier

    if (idx < N - M + 1) {

      float residues = 0;
      float avg = avgs[idx];
      float std = stds[idx];

      for (int i = 0; i < M; ++i) {
        float value = (cache[lid % num_threads + i] - avg) / std;
        float lower = value - lower_bound[i];
        float upper = value - upper_bound[i];

        residues += upper * upper * (upper > 0) + lower * lower * (lower < 0);
      }

      lb_keogh[idx] = residues;
    }
  }
}

int main(int argc, char* argv[]) {

  if (argc != 4) {
    printf("Usage: ./%s <query length> <subject length> <repeat>\n", argv[0]);
    return 1;
  }

  const int M = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  printf("Query length = %d\n", M);
  printf("Subject length = %d\n", N);

  float *subject = (float*) malloc (sizeof(float)*N);
  float *lower = (float*) malloc (sizeof(float)*N);
  float *upper = (float*) malloc (sizeof(float)*N);
  float *lb = (float*) malloc (sizeof(float)*(N-M+1));
  float *lb_h = (float*) malloc (sizeof(float)*(N-M+1));
  float *avgs = (float*) malloc (sizeof(float)*(N-M+1));
  float *stds = (float*) malloc (sizeof(float)*(N-M+1));

  srand(123);
  for (int i = 0; i < N; ++i) subject[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < N-M+1; ++i) avgs[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < N-M+1; ++i) stds[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < M; ++i) upper[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < M; ++i) lower[i] = (float)rand() / (float)RAND_MAX;

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    lb_keogh(subject, avgs, stds, lb, lower, upper, M, N);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  reference(subject, avgs, stds, lb_h, lower, upper, M, N);
  bool ok = true;
  for (int i = 0; i < N-M+1; i++) {
    if (fabsf(lb[i] - lb_h[i]) > 1e-3f) {
      printf("%d %f %f\n", i, lb[i], lb_h[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(lb);
  free(lb_h);
  free(avgs);
  free(stds);
  free(subject);
  free(lower);
  free(upper);
  return 0;
}
```