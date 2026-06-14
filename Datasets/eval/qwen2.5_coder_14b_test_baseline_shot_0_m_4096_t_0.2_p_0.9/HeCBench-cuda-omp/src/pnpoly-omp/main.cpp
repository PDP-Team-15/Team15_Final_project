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
#include <random>
#include <chrono>
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

  // No need to allocate device memory in OpenMP

  #pragma omp parallel for
  for (int i = 0; i < nPoints; i++) {
    bitmap_ref[i] = 0;
    bitmap_opt[i] = 0;
  }

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < nPoints; j++) {
      for (int k = 0; k < vertices; k++) {
        pnpoly_base(bitmap_ref, point, vertex, nPoints, j, k);
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_base): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < nPoints; j++) {
      for (int k = 0; k < vertices; k++) {
        pnpoly_opt<1>(bitmap_opt, point, vertex, nPoints, j, k);
      }
    }
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<1>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < nPoints; j++) {
      for (int k = 0; k < vertices; k++) {
        pnpoly_opt<2>(bitmap_opt, point, vertex, nPoints, j, k);
      }
    }
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<2>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < nPoints; j++) {
      for (int k = 0; k < vertices; k++) {
        pnpoly_opt<4>(bitmap_opt, point, vertex, nPoints, j, k);
      }
    }
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<4>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < nPoints; j++) {
      for (int k = 0; k < vertices; k++) {
        pnpoly_opt<8>(bitmap_opt, point, vertex, nPoints, j, k);
      }
    }
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<8>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < nPoints; j++) {
      for (int k = 0; k < vertices; k++) {
        pnpoly_opt<16>(bitmap_opt, point, vertex, nPoints, j, k);
      }
    }
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<16>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < nPoints; j++) {
      for (int k = 0; k < vertices; k++) {
        pnpoly_opt<32>(bitmap_opt, point, vertex, nPoints, j, k);
      }
    }
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<32>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < nPoints; j++) {
      for (int k = 0; k < vertices; k++) {
        pnpoly_opt<64>(bitmap_opt, point, vertex, nPoints, j, k);
      }
    }
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<64>): %f (s)\n", (time * 1e-9f) / repeat);

  int error = memcmp(bitmap_opt, bitmap_ref, nPoints*sizeof(int)); 

  int checksum = 0;
  for (int i = 0; i < nPoints; i++) checksum += bitmap_opt[i];
  printf("Checksum: %d\n", checksum);

  printf("%s\n", error ? "FAIL" : "PASS");

  free(vertex);
  free(point);
  free(bitmap_ref);
  free(bitmap_opt);
  return 0;
}