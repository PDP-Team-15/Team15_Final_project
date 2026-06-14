Here's the equivalent code in OpenMP:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>

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

  float2 *point = (float2*) malloc (sizeof(float2) * nPoints);
  for (int i = 0; i < nPoints; i++) {
    point[i].x = (float)rand() / RAND_MAX * 2.f - 1.f;
    point[i].y = (float)rand() / RAND_MAX * 2.f - 1.f;
  }

  float2 *vertex = (float2*) malloc (vertices * sizeof(float2));
  for (int i = 0; i < vertices; i++) {
    float t = (float)rand() / RAND_MAX * 2.f * 3.14159265358979323846;
    vertex[i].x = cosf(t);
    vertex[i].y = sinf(t);
  }

  int *bitmap_ref = (int*) malloc (nPoints * sizeof(int));
  int *bitmap_opt = (int*) malloc (nPoints * sizeof(int));

  float2 *d_point;
  float2 *d_vertex;
  int *d_bitmap_ref, *d_bitmap_opt;

  // No need for dynamic memory allocation in OpenMP

  int num_threads = omp_get_max_threads();
  int num_blocks = (nPoints + num_threads - 1) / num_threads;

  clock_t start, end;
  double time;

  for (int i = 0; i < repeat; i++) {
    start = clock();
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_blocks; i++) {
      int block_start = i * num_threads;
      int block_end = (i + 1) * num_threads;
      if (block_end > nPoints) block_end = nPoints;

      for (int j = block_start; j < block_end; j++) {
        int point_index = j;
        int vertex_index = 0;
        int inside = 0;
        int i, j, c = 0;

        for (i = 0, j = 0; i < vertices; i++) {
          while (j < vertices && vertex[j].x < point[point_index].x) {
            j++;
          }

          if (j > 0 && j < vertices && vertex[j - 1].x <= point[point_index].x && point[point_index].x < vertex[j].x) {
            c += 1;
            if ((point[point_index].y > vertex[j - 1].y) != (point[point_index].y > vertex[j].y)) {
              inside += 1;
            }
          }
        }

        bitmap_ref[point_index] = (inside & 1);
      }
    }
    end = clock();
    time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Average kernel execution time (pnpoly_base): %f (s)\n", time / repeat);
  }

  for (int i = 0; i < repeat; i++) {
    start = clock();
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_blocks; i++) {
      int block_start = i * num_threads;
      int block_end = (i + 1) * num_threads;
      if (block_end > nPoints) block_end = nPoints;

      for (int j = block_start; j < block_end; j++) {
        int point_index = j;
        int vertex_index = 0;
        int inside = 0;
        int i, j, c = 0;

        for (i = 0, j = 0; i < vertices; i++) {
          while (j < vertices && vertex[j].x < point[point_index].x) {
            j++;
          }

          if (j > 0 && j < vertices && vertex[j - 1].x <= point[point_index].x && point[point_index].x < vertex[j].x) {
            c += 1;
            if ((point[point_index].y > vertex[j - 1].y) != (point[point_index].y > vertex[j].y)) {
              inside += 1;
            }
          }
        }

        bitmap_opt[point_index] = (inside & 1);
      }
    }
    end = clock();
    time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Average kernel execution time (pnpoly_opt<1>): %f (s)\n", time / repeat);
  }

  for (int i = 0; i < repeat; i++) {
    start = clock();
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_blocks; i++) {
      int block_start = i * num_threads;
      int block_end = (i + 1) * num_threads;
      if (block_end > nPoints) block_end = nPoints;

      for (int j = block_start; j < block_end; j++) {
        int point_index = j;
        int vertex_index = 0;
        int inside = 0;
        int i, j, c = 0;

        for (i = 0, j = 0; i < vertices; i++) {
          while (j < vertices && vertex[j].x < point[point_index].x) {
            j++;
          }

          if (j > 0 && j < vertices && vertex[j - 1].x <= point[point_index].x && point[point_index].x < vertex[j].x) {
            c += 1;
            if ((point[point_index].y > vertex[j - 1].y) != (point[point_index].y > vertex[j].y)) {
              inside += 1;
            }
          }
        }

        bitmap_opt[point_index] = (inside & 1);
      }
    }
    end = clock();
    time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Average kernel execution time (pnpoly_opt<2>): %f (s)\n", time / repeat);
  }

  for (int i = 0; i < repeat; i++) {
    start = clock();
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_blocks; i++) {
      int block_start = i * num_threads;
      int block_end = (i + 1) * num_threads;
      if (block_end > nPoints) block_end = nPoints;

      for (int j = block_start; j < block_end; j++) {
        int point_index = j;
        int vertex_index = 0;
        int inside = 0;
        int i, j, c = 0;

        for (i = 0, j = 0; i < vertices; i++) {
          while (j < vertices && vertex[j].x < point[point_index].x) {
            j++;
          }

          if (j > 0 && j < vertices && vertex[j - 1].x <= point[point_index].x && point[point_index].x < vertex[j].x) {
            c += 1;
            if ((point[point_index].y > vertex[j - 1].y) != (point[point_index].y > vertex[j].y)) {
              inside += 1;
            }
          }
        }

        bitmap_opt[point_index] = (inside & 1);
      }
    }
    end = clock();
    time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Average kernel execution time (pnpoly_opt<4>): %f (s)\n", time / repeat);
  }

  for (int i = 0; i < repeat; i++) {
    start = clock();
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_blocks; i++) {
      int block_start = i * num_threads;
      int block_end = (i + 1) * num_threads;
      if (block_end > nPoints) block_end = nPoints;

      for (int j = block_start; j < block_end; j++) {
        int point_index = j;
        int vertex_index = 0;
        int inside = 0;
        int i, j, c = 0;

        for (i = 0, j = 0; i < vertices; i++) {
          while (j < vertices && vertex[j].x < point[point_index].x) {
            j++;
          }

          if (j > 0 && j < vertices && vertex[j - 1].x <= point[point_index].x && point[point_index].x < vertex[j].x) {
            c += 1;
            if ((point[point_index].y > vertex[j - 1].y) != (point[point_index].y > vertex[j].y)) {
              inside += 