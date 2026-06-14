Here's the equivalent OpenMP code for the given CUDA code:

```c
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <chrono>
#include "reference.h"

void lb_keogh(float *subject, float *avgs, float *stds, float *lb_keogh, float *lower_bound, float *upper_bound, const int M, const int N) {
  int tid, num_threads;
  float cache[M + 64];

  #pragma omp parallel for default(none) shared(subject, avgs, stds, lb_keogh, lower_bound, upper_bound, M, N) num_threads(256)
  for (int i = 0; i < N; i += 256) {
    for (int k = 0; k < M + 64; k++) {
      if (i + k < N) cache[k] = subject[i + k];
    }
    #pragma omp barrier

    for (int idx = i; idx < i + 256 && idx < N - M + 1; idx++) {
      float residues = 0;
      float avg = avgs[idx];
      float std = stds[idx];

      for (int i = 0; i < M; ++i) {
        float value = (cache[i] - avg) / std;
        float lower = value - lower_bound[i];
        float upper = value - upper_bound[i];

        residues += upper*upper*(upper > 0) + lower*lower*(lower < 0);
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

  for (int i = 0; i < repeat; i++) {
    lb_keogh(subject, avgs, stds, lb, lower, upper, M, N);
  }

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

Note that OpenMP does not have a direct equivalent to CUDA's shared memory. However, we can use a shared array to achieve similar functionality. In this code, we use a shared array `cache` to store the subject data for each thread block. The size of the shared array is set to `M + 64` to accommodate the maximum number of threads in a block (256) and the query length `M`.

Also, note that OpenMP does not have a direct equivalent to CUDA's `cudaDeviceSynchronize()` function. However, we can use the `omp barrier` directive to ensure that all threads in a block have completed their tasks before proceeding to the next iteration.

The `num_threads(256)` clause is used to specify the number of threads in each thread block. This is equivalent to setting the `blockSize` variable in the CUDA code.

The `default(none)` clause is used to specify that all variables should be explicitly declared as shared or private. This is equivalent to setting the `__restrict__` keyword in the CUDA code.

The `shared(subject, avgs, stds, lb_keogh, lower_bound, upper_bound, M, N)` clause is used to specify the shared variables for each thread block. This is equivalent to passing the variables as arguments to the CUDA kernel function.

The `omp parallel for` directive is used to parallelize the loop over the subject data. This is equivalent to launching a CUDA kernel function.

The `omp barrier` directive is used to ensure that all threads in a block have completed their tasks before proceeding to the next iteration. This is equivalent to calling `__syncthreads()` in the CUDA code.

The `omp for` directive is used to parallelize the loop over the subject data. This is equivalent to launching a CUDA kernel function.

The `omp parallel` directive is used to specify the number of threads in each thread block. This is equivalent to setting the `blockSize` variable in the CUDA code.

The `omp shared` directive is used to specify the shared variables for each thread block. This is equivalent to passing the variables as arguments to the CUDA kernel function.

The `omp private` directive is used to specify the private variables for each thread. This is equivalent to declaring the variables as `__device__` in the CUDA code.

The `omp firstprivate` directive is used to specify the firstprivate variables for each thread. This is equivalent to declaring the variables as `__device__` in the CUDA code.

The `omp reduction` directive is used to specify the reduction operation for each thread. This is equivalent to using the `__shared__` keyword in the CUDA code.

The `omp critical` directive is used to specify the critical section for each thread. This is equivalent to using the `__syncthreads()` function in the CUDA code.

The `omp atomic` directive is used to specify the atomic operation for each thread. This is equivalent to using the `__syncthreads()` function in the CUDA code.

Note that the OpenMP code may not be exactly equivalent to the CUDA code in terms of performance and functionality. However, it should provide a similar level of parallelism and scalability.