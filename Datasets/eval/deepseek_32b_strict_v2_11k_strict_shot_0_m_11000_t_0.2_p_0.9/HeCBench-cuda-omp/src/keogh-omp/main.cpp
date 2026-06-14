#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <omp.h>

void lb_keogh(const float *subject,
              const float *avgs,
              const float *stds,
              float *lb_keogh,
              const float *lower_bound,
              const float *upper_bound,
              const int M,
              const int N) {
  int num_threads = omp_get_max_threads();
  float *cache = (float *)malloc(num_threads * (M + num_threads) * sizeof(float));

  #pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    float *thread_cache = cache + thread_id * (M + num_threads);

    for (int k = thread_id; k < N; k += num_threads) {
      if (k < N) thread_cache[k] = subject[k];
    }

    #pragma omp barrier

    int start = (N - M + 1) * thread_id / num_threads;
    int end = (N - M + 1) * (thread_id + 1) / num_threads;

    for (int i = start; i < end; ++i) {
      if (i < N - M + 1) {
        float residues = 0;
        float avg = avgs[i];
        float std = stds[i];

        for (int j = 0; j < M; ++j) {
          float value = (thread_cache[i + j] - avg) / std;
          float lower = value - lower_bound[j];
          float upper = value - upper_bound[j];

          residues += upper * upper * (upper > 0) + lower * lower * (lower < 0);
        }

        lb_keogh[i] = residues;
      }
    }
  }

  free(cache);
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: ./%s <query length> <subject length> <repeat>\n", argv[0]);
    return 1;
  }

  const int M = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  printf("Query length = %d\n", M);
  printf("Subject length = %d\n", N);

  float *subject = (float *)malloc(N * sizeof(float));
  float *lower = (float *)malloc(N * sizeof(float));
  float *upper = (float *)malloc(N * sizeof(float));
  float *lb = (float *)malloc((N - M + 1) * sizeof(float));
  float *lb_h = (float *)malloc((N - M + 1) * sizeof(float));
  float *avgs = (float *)malloc((N - M + 1) * sizeof(float));
  float *stds = (float *)malloc((N - M + 1) * sizeof(float));

  srand(123);
  for (int i = 0; i < N; ++i) subject[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < N - M + 1; ++i) avgs[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < N - M + 1; ++i) stds[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < M; ++i) upper[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < M; ++i) lower[i] = (float)rand() / (float)RAND_MAX;

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; ++i) {
    lb_keogh(subject, avgs, stds, lb, lower, upper, M, N);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  reference(subject, avgs, stds, lb_h, lower, upper, M, N);
  bool ok = true;
  for (int i = 0; i < N - M + 1; ++i) {
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