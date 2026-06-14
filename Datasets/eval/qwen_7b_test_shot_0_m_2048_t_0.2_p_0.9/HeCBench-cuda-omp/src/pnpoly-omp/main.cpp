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

#define VERTICES 600
#define THREADS_PER_BLOCK 256

#include "kernel.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: ./%s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  const int nPoints = 2e7;
  const int vertices = VERTICES;

  std::default_random_engine rng(123);
  std::normal_distribution<float> distribution(0, 1);

  float2 *point = (float2*)malloc(sizeof(float2) * nPoints);
  for (int i = 0; i < nPoints; i++) {
    point[i].x = distribution(rng);
    point[i].y = distribution(rng);
  }

  float2 *vertex = (float2*)malloc(vertices * sizeof(float2));
  for (int i = 0; i < vertices; i++) {
    float t = distribution(rng) * 2.f * M_PI;
    vertex[i].x = cosf(t);
    vertex[i].y = sinf(t);
  }

  int *bitmap_ref = (int*)malloc(nPoints * sizeof(int));
  int *bitmap_opt = (int*)malloc(nPoints * sizeof(int));

  #pragma omp parallel for num_threads(THREADS_PER_BLOCK)
  for (int i = 0; i < nPoints; i++) {
    bitmap_ref[i] = 0;
    bitmap_opt[i] = 0;
  }

  #pragma omp parallel for collapse(2) num_threads(THREADS_PER_BLOCK)
  for (int r = 0; r < repeat; r++) {
    for (int i = 0; i < nPoints; i++) {
      pnpoly_base(bitmap_ref, &point[i], vertex, nPoints);
    }
  }

  #pragma omp parallel for collapse(2) num_threads(THREADS_PER_BLOCK)
  for (int r = 0; r < repeat; r++) {
    for (int bits = 1; bits <= 64; bits *= 2) {
      for (int i = 0; i < nPoints; i++) {
        pnpoly_opt<bits>(bitmap_opt, &point[i], vertex, nPoints);
      }
    }
  }

  int error = memcmp(bitmap_opt, bitmap_ref, nPoints * sizeof(int));

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