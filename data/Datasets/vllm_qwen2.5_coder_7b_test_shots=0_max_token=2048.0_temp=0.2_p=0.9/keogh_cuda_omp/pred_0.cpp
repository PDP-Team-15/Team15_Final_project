Here is the translation of the CUDA code to OpenMP:

```c
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "reference.h"

void lb_keogh(const float *subject,
              const float *avgs,
              const float *stds, 
                    float *lb_keogh,
              const float *lower_bound,
              const float *upper_bound,
              const int M,
              const int N) 
{
  #pragma omp parallel for shared(subject, avgs, stds, lb_keogh, lower_bound, upper_bound, M, N)
  for (int idx = 0; idx < N-M+1; ++idx) {
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

  const int blocks = 256;
  const int grids = (N-M+1 + blocks - 1) / blocks;
  int smem_size = (M+blocks)*sizeof(float);

  double start = omp_get_wtime();

  for (int i = 0; i < repeat; i++) {
    lb_keogh(subject, avgs, stds, lb, lower, upper, M, N);
  }

  double end = omp_get_wtime();
  double time = (end - start) / repeat;
  printf("Average kernel execution time: %f (s)\n", time);

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

### Explanation:
1. **Kernel Function (`lb_keogh`)**:
   - The `__global__` keyword is replaced with `#pragma omp parallel for`.
   - Shared memory and synchronization mechanisms specific to CUDA are removed as they are not needed in OpenMP.
   - The loop over `idx` is now a parallel region.

2. **Main Function**:
   - CUDA-specific functions like `cudaMalloc`, `cudaMemcpy`, and `cudaDeviceSynchronize` are replaced with their OpenMP equivalents where applicable.
   - The timing mechanism is changed from `std::chrono` to `omp_get_wtime()` for better compatibility with OpenMP.

This translation ensures that the code runs on systems that support OpenMP without requiring CUDA.