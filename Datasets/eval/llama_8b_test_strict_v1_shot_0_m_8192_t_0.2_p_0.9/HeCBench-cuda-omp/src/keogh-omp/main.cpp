#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <chrono>
#include "reference.h"

void lb_keogh(const float *__restrict__ subject,
              const float *__restrict__ avgs,
              const float *__restrict__ stds, 
                    float *__restrict__ lb_keogh,
              const float *__restrict__ lower_bound,
              const float *__restrict__ upper_bound,
              const int M,
              const int N) 
{
  int lid = omp_get_thread_num();
  int num_threads = omp_get_num_threads();
  int idx = lid;

  for (int k = lid; k < num_threads + M; k += num_threads)
    if (idx + k < N) subject[idx + k] = subject[idx + k];

  __sync_threads();

  if (idx < N-M+1) {

    float residues = 0;
    float avg = avgs[idx];
    float std = stds[idx];

    for (int i = 0; i < M; ++i) {
      float value = (subject[idx+i] - avg) / std;
      float lower = value - lower_bound[i];
      float upper = value - upper_bound[i];

      residues += upper*upper*(upper > 0) + lower*lower*(lower < 0);
    }

    lb_keogh[idx] = residues;
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
  float *lower = (float*) malloc (sizeof(float)*M);
  float *upper = (float*) malloc (sizeof(float)*M);
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

  float *lb_keogh = (float*) malloc (sizeof(float)*(N-M+1));

  int num_threads = omp_get_max_threads();
  int num_blocks = (N-M+1 + num_threads - 1) / num_threads;

  #pragma omp parallel for num_threads(num_threads) default(none) \
    shared(subject, avgs, stds, lb_keogh, lower, upper, M, N) \
    private(lid, idx, residues, avg, std, i, value, lower, upper)
  for (int idx = 0; idx < N-M+1; idx++) {
    float residues = 0;
    float avg = avgs[idx];
    float std = stds[idx];

    for (int i = 0; i < M; ++i) {
      float value = (subject[idx+i] - avg) / std;
      float lower = value - lower[i];
      float upper = value - upper[i];

      residues += upper*upper*(upper > 0) + lower*lower*(lower < 0);
    }

    lb_keogh[idx] = residues;
  }

  #pragma omp parallel for num_threads(num_threads) default(none) \
    shared(lb_keogh, subject, avgs, stds, lower, upper, M, N) \
    private(lid, idx, avg, std, i, value, lower, upper)
  for (int idx = 0; idx < N-M+1; idx++) {
    float avg = avgs[idx];
    float std = stds[idx];

    for (int i = 0; i < M; ++i) {
      float value = (subject[idx+i] - avg) / std;
      float lower = value - lower[i];
      float upper = value - upper[i];

      lb_keogh[idx] += upper*upper*(upper > 0) + lower*lower*(lower < 0);
    }
  }

  reference(subject, avgs, stds, lb_h, lower, upper, M, N);
  bool ok = true;
  for (int i = 0; i < N-M+1; i++) {
    if (fabsf(lb_keogh[i] - lb_h[i]) > 1e-3f) {
      printf("%d %f %f\n", i, lb_keogh[i], lb_h[i]);
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
  free(lb_keogh);
  return 0;
}