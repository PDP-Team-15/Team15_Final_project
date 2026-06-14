```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define VERTICES 600
#define BLOCK_SIZE_X 256

#include "kernel.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: ./%s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  const int nPoints = 20000000;
  const int vertices = VERTICES;

  std::default_random_engine rng (123);
  std::normal_distribution<float> distribution(0, 1);

  float2 *point = (float2*) malloc (sizeof(float2) * nPoints);
  for (int i = 0; i < nPoints; i++) {
    point[i].x = distribution(rng);
    point[i].y = distribution(rng);
  }

  float2 *vertex = (float2*) malloc (vertices * sizeof(float2));
  for (int i = 0; i < vertices; i++) {
    float t = distribution(rng) * 2.f * M_PI;
    vertex[i].x = cosf(t);
    vertex[i].y = sinf(t);
  }

  int *bitmap_ref = (int*) malloc (nPoints * sizeof(int));
  int *bitmap_opt = (int*) malloc (nPoints * sizeof(int));

  float2 *point_omp;
  float2 *vertex_omp;
  int *bitmap_ref_omp, *bitmap_opt_omp;

  omp_lock_t lock;

  omp_init_lock(&lock);

  #pragma omp parallel for
  for (int i = 0; i < nPoints; i++) {
    bitmap_ref[i] = 0;
    bitmap_opt[i] = 0;
  }

  point_omp = (float2*) malloc (sizeof(float2) * nPoints);
  vertex_omp = (float2*) malloc (vertices * sizeof(float2));
  bitmap_ref_omp = (int*) malloc (nPoints * sizeof(int));
  bitmap_opt_omp = (int*) malloc (nPoints * sizeof(int));

  #pragma omp parallel for
  for (int i = 0; i < nPoints; i++) {
    point_omp[i].x = point[i].x;
    point_omp[i].y = point[i].y;
  }

  #pragma omp parallel for
  for (int i = 0; i < vertices; i++) {
    float t = distribution(rng) * 2.f * M_PI;
    vertex_omp[i].x = cosf(t);
    vertex_omp[i].y = sinf(t);
  }

  int num_threads = omp_get_max_threads();
  int thread_id = omp_get_thread_num();

  int grid_size = (nPoints + num_threads - 1) / num_threads;
  int block_size = num_threads;

  #pragma omp parallel
  {
    #pragma omp for
    for (int i = 0; i < repeat; i++) {
      pnpoly_base(bitmap_ref_omp, point_omp, vertex_omp, nPoints, grid_size, block_size, thread_id);
    }
  }

  #pragma omp parallel
  {
    #pragma omp for
    for (int i = 0; i < repeat; i++) {
      pnpoly_opt<1>(bitmap_opt_omp, point_omp, vertex_omp, nPoints, grid_size, block_size, thread_id);
    }
  }

  #pragma omp parallel
  {
    #pragma omp for
    for (int i = 0; i < repeat; i++) {
      pnpoly_opt<2>(bitmap_opt_omp, point_omp, vertex_omp, nPoints, grid_size, block_size, thread_id);
    }
  }

  #pragma omp parallel
  {
    #pragma omp for
    for (int i = 0; i < repeat; i++) {
      pnpoly_opt<4>(bitmap_opt_omp, point_omp, vertex_omp, nPoints, grid_size, block_size, thread_id);
    }
  }

  #pragma omp parallel
  {
    #pragma omp for
    for (int i = 0; i < repeat; i++) {
      pnpoly_opt<8>(bitmap_opt_omp, point_omp, vertex_omp, nPoints, grid_size, block_size, thread_id);
    }
  }

  #pragma omp parallel
  {
    #pragma omp for
    for (int i = 0; i < repeat; i++) {
      pnpoly_opt<16>(bitmap_opt_omp, point_omp, vertex_omp, nPoints, grid_size, block_size, thread_id);
    }
  }

  #pragma omp parallel
  {
    #pragma omp for
    for (int i = 0; i < repeat; i++) {
      pnpoly_opt<32>(bitmap_opt_omp, point_omp, vertex_omp, nPoints, grid_size, block_size, thread_id);
    }
  }

  #pragma omp parallel
  {
    #pragma omp for
    for (int i = 0; i < repeat; i++) {
      pnpoly_opt<64>(bitmap_opt_omp, point_omp, vertex_omp, nPoints, grid_size, block_size, thread_id);
    }
  }

  #pragma omp parallel
  {
    #pragma omp for
    for (int i = 0; i < nPoints; i++) {
      bitmap_ref[i] = bitmap_ref_omp[i];
    }
  }

  int error = memcmp(bitmap_opt, bitmap_ref, nPoints*sizeof(int)); 
  
  int checksum = 0;
  for (int i = 0; i < nPoints; i++) checksum += bitmap_opt[i];
  printf("Checksum: %d\n", checksum);

  printf("%s\n", error ? "FAIL" : "PASS");

  free(point);
  free(vertex);
  free(bitmap_ref);
  free(bitmap_opt);
  return 0;
}
```

Note: The `pnpoly_base` and `pnpoly_opt` functions are assumed to be defined in the `kernel.h` header file. The `float2` type is assumed to be a struct with `x` and `y` members. The `omp_lock_t` type is used for synchronization. The `omp_init_lock` and `omp_destroy_lock` functions are used to initialize and destroy the lock. The `omp_set_lock` and `omp_unset_lock` functions are used to acquire and release the lock. The `omp_get_max_threads` and `omp_get_thread_num` functions are used to get the maximum number of threads and the current thread number. The `omp_set_num_threads` function is used to set the number of threads. The `omp parallel` and `omp for` directives are used to parallelize the code. The `grid_size` and `block_size` variables are used to specify the grid and block sizes for the parallel regions. The `thread_id` variable is used to get the current thread ID.