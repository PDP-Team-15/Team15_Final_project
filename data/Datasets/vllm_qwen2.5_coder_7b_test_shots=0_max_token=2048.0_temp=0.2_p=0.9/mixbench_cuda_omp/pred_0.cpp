Here is the translation of the CUDA code to OpenMP:

```c
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define VECTOR_SIZE (8*1024*1024)
#define granularity (8)
#define fusion_degree (4)
#define seed 0.1f

void benchmark_func(float *data,
                   const int compute_iterations)
{
  const unsigned int blockSize = 256;
  const int stride = blockSize;
  int idx = omp_get_thread_num() * blockSize * granularity;
  const int big_stride = omp_get_num_threads() * blockSize * granularity;

  float tmps[granularity];
  for(int k=0; k<fusion_degree; k++) {
    #pragma omp simd
    for(int j=0; j<granularity; j++) {
      tmps[j] = data[idx+j*stride+k*big_stride];

      for(int i=0; i<compute_iterations; i++)
        tmps[j] = tmps[j]*tmps[j]+seed;
    }

    float sum = 0.f;
    #pragma omp simd reduction(+:sum)
    for(int j=0; j<granularity; j+=2)
      sum += tmps[j]*tmps[j+1];

    #pragma omp simd
    for(int j=0; j<granularity; j++)
      data[idx+k*big_stride] = sum;
  }
}

void mixbenchGPU(long size, int repeat) {
  const char *benchtype = "compute with global memory (block strided)";
  printf("Trade-off type:%s\n", benchtype);
  float *cd = (float*) malloc (size*sizeof(float));
  for (int i = 0; i < size; i++) cd[i] = 0;

  const long reduced_grid_size = size/granularity/128;
  const int block_dim = 256;
  const int grid_dim = reduced_grid_size/block_dim;

  // No need for device allocation and copy in OMP

  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel for collapse(2)
    for (int k=0; k<fusion_degree; k++) {
      for (int j=0; j<granularity; j++) {
        benchmark_func(cd, i);
      }
    }
  }

  clock_t start = clock();
  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel for collapse(2)
    for (int k=0; k<fusion_degree; k++) {
      for (int j=0; j<granularity; j++) {
        benchmark_func(cd, i);
      }
    }
  }
  clock_t end = clock();
  double time = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Total kernel execution time: %f (s)\n", time);

  // No need for device copy back in OMP

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
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  unsigned int datasize = VECTOR_SIZE*sizeof(float);

  printf("Buffer size: %dMB\n", datasize/(1024*1024));

  mixbenchGPU(VECTOR_SIZE, repeat);

  return 0;
}
```

### Explanation:
1. **Global Memory Access**: In CUDA, global memory access is handled by the GPU. In OMP, we simulate this using a single array `cd` that is accessed by all threads.
2. **Block Size and Grid Size**: The block size and grid size are defined as constants. In OMP, we use `omp_get_thread_num()` and `omp_get_num_threads()` to manage thread indices.
3. **Kernel Execution**: The kernel execution is simulated using nested loops in OMP. The `#pragma omp parallel for collapse(2)` directive is used to parallelize the outer two loops.
4. **Timing**: The timing is done using `clock()` instead of `std::chrono`.
5. **Memory Management**: There is no need for device memory allocation and copy operations in OMP.

This translation should provide similar performance characteristics to the original CUDA code while being executed on a CPU using OpenMP.