#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <chrono>
#include <omp.h>
#include "kernel.h"
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
  const int nPoints = 2e7;
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

  float2 *point_host;
  float2 *vertex_host;
  int *bitmap_ref_host, *bitmap_opt_host;

  point_host = point;
  vertex_host = vertex;
  bitmap_ref_host = bitmap_ref;
  bitmap_opt_host = bitmap_opt;

  int num_threads = omp_get_max_threads();
  int num_blocks = (nPoints + num_threads - 1) / num_threads;

  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel for num_threads(num_threads)
    for (int j = 0; j < num_blocks; j++) {
      int block_start = j * num_threads;
      int block_end = (j + 1) * num_threads;
      if (block_end > nPoints) block_end = nPoints;

      #pragma omp for
      for (int k = block_start; k < block_end; k++) {
        pnpoly_base(bitmap_ref_host[k], point_host + k, vertex_host, nPoints);
      }
    }
  }

  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel for num_threads(num_threads)
    for (int j = 0; j < num_blocks; j++) {
      int block_start = j * num_threads;
      int block_end = (j + 1) * num_threads;
      if (block_end > nPoints) block_end = nPoints;

      #pragma omp for
      for (int k = block_start; k < block_end; k++) {
        pnpoly_opt<1>(bitmap_opt_host[k], point_host + k, vertex_host, nPoints);
      }
    }
  }

  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel for num_threads(num_threads)
    for (int j = 0; j < num_blocks; j++) {
      int block_start = j * num_threads;
      int block_end = (j + 1) * num_threads;
      if (block_end > nPoints) block_end = nPoints;

      #pragma omp for
      for (int k = block_start; k < block_end; k++) {
        pnpoly_opt<2>(bitmap_opt_host[k], point_host + k, vertex_host, nPoints);
      }
    }
  }

  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel for num_threads(num_threads)
    for (int j = 0; j < num_blocks; j++) {
      int block_start = j * num_threads;
      int block_end = (j + 1) * num_threads;
      if (block_end > nPoints) block_end = nPoints;

      #pragma omp for
      for (int k = block_start; k < block_end; k++) {
        pnpoly_opt<4>(bitmap_opt_host[k], point_host + k, vertex_host, nPoints);
      }
    }
  }

  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel for num_threads(num_threads)
    for (int j = 0; j < num_blocks; j++) {
      int block_start = j * num_threads;
      int block_end = (j + 1) * num_threads;
      if (block_end > nPoints) block_end = nPoints;

      #pragma omp for
      for (int k = block_start; k < block_end; k++) {
        pnpoly_opt<8>(bitmap_opt_host[k], point_host + k, vertex_host, nPoints);
      }
    }
  }

  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel for num_threads(num_threads)
    for (int j = 0; j < num_blocks; j++) {
      int block_start = j * num_threads;
      int block_end = (j + 1) * num_threads;
      if (block_end > nPoints) block_end = nPoints;

      #pragma omp for
      for (int k = block_start; k < block_end; k++) {
        pnpoly_opt<16>(bitmap_opt_host[k], point_host + k, vertex_host, nPoints);
      }
    }
  }

  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel for num_threads(num_threads)
    for (int j = 0; j < num_blocks; j++) {
      int block_start = j * num_threads;
      int block_end = (j + 1) * num_threads;
      if (block_end > nPoints) block_end = nPoints;

      #pragma omp for
      for (int k = block_start; k < block_end; k++) {
        pnpoly_opt<32>(bitmap_opt_host[k], point_host + k, vertex_host, nPoints);
      }
    }
  }

  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel for num_threads(num_threads)
    for (int j = 0; j < num_blocks; j++) {
      int block_start = j * num_threads;
      int block_end = (j + 1) * num_threads;
      if (block_end > nPoints) block_end = nPoints;

      #pragma omp for
      for (int k = block_start; k < block_end; k++) {
        pnpoly_opt<64>(bitmap_opt_host[k], point_host + k, vertex_host, nPoints);
      }
    }
  }

  int error = memcmp(bitmap_opt_host, bitmap_ref_host, nPoints*sizeof(int)); 
  
  int checksum = 0;
  for (int i = 0; i < nPoints; i++) checksum += bitmap_opt_host[i];
  printf("Checksum: %d\n", checksum);

  printf("%s\n", error ? "FAIL" : "PASS");

  free(vertex);
  free(point);
  free(bitmap_ref);
  free(bitmap_opt);
  return 0;
}