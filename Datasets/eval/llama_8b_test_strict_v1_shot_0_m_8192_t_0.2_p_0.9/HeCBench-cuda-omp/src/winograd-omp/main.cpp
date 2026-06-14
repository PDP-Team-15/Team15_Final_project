#include <chrono>
#include <omp.h>
#include "utils.h"
#include <omp.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAP_SIZE 256
#define DIM_LOCAL_WORK_GROUP_X 16
#define DIM_LOCAL_WORK_GROUP_Y 16

#define DATA_TYPE float

#define ceil(x) ((x) > 0 ? (int)ceil((x)) : (int)floor((x)))

void WinogradConv2D_2x2_omp(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, size_t cpu_global_size[2]) {
  int tile_n = (MAP_SIZE - 2 + 1) / 2;
  int global_offset[2] = {cpu_global_size[0], 0};
  int local_work_group_size[2] = {DIM_LOCAL_WORK_GROUP_X, DIM_LOCAL_WORK_GROUP_Y};

  #pragma omp parallel for
  for (int i = 0; i < tile_n; i++) {
    for (int j = 0; j < tile_n; j++) {
      int tile_i = i + global_offset[0];
      int tile_j = j + global_offset[1];

      DATA_TYPE input_tile[4][4], tmp_tile[4][4], transformed_tile[4][4];
      for (int k = 0; k < 4; k++) {
        for (int l = 0; l < 4; l++) {
          int x = 2 * tile_i + k;
          int y = 2 * tile_j + l;
          if (x >= MAP_SIZE || y >= MAP_SIZE) {
            input_tile[k][l] = 0;
            continue;
          }
          input_tile[k][l] = A[x * MAP_SIZE + y];
        }
      }

      for (int l = 0; l < 4; l++) {
        tmp_tile[0][l] = input_tile[0][l] - input_tile[2][l];
        tmp_tile[1][l] = input_tile[1][l] + input_tile[2][l];
        tmp_tile[2][l] = -input_tile[1][l] + input_tile[2][l];
        tmp_tile[3][l] = input_tile[1][l] - input_tile[3][l];
      }

      for (int k = 0; k < 4; k++) {
        transformed_tile[k][0] = tmp_tile[k][0] - tmp_tile[k][2];
        transformed_tile[k][1] = tmp_tile[k][1] + tmp_tile[k][2];
        transformed_tile[k][2] = -tmp_tile[k][1] + tmp_tile[k][2];
        transformed_tile[k][3] = tmp_tile[k][1] - tmp_tile[k][3];
      }

      DATA_TYPE multiplied_tile[4][4];
      for (int k = 0; k < 4; k++) {
        for (int l = 0; l < 4; l++) {
          multiplied_tile[k][l] = transformed_tile[k][l] * C[k * 4 + l];
        }
      }

      DATA_TYPE tmp_tile_1[2][4], final_tile[2][2];

      for (int l = 0; l < 4; l++) {
        tmp_tile_1[0][l] = multiplied_tile[0][l] + multiplied_tile[1][l] + multiplied_tile[2][l];
        tmp_tile_1[1][l] = multiplied_tile[1][l] - multiplied_tile[2][l] - multiplied_tile[3][l];
      }

      for (int k = 0; k < 2; k++) {
        final_tile[k][0] = tmp_tile_1[k][0] + tmp_tile_1[k][1] + tmp_tile_1[k][2];
        final_tile[k][1] = tmp_tile_1[k][1] - tmp_tile_1[k][2] - tmp_tile_1[k][3];
      }

      for (int k = 0; k < 2; k++) {
        for (int l = 0; l < 2; l++) {
          int x = 2 * tile_i + k;
          int y = 2 * tile_j + l;
          if (x >= MAP_SIZE - 2 || y >= MAP_SIZE - 2) {
            continue;
          }
          B[x * (MAP_SIZE - 2) + y] = final_tile[k][l];
        }
      }
    }
  }
}

int main(int argc, char* argv[]) {
  double start = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

  DATA_TYPE *A = (DATA_TYPE*)malloc(MAP_SIZE * MAP_SIZE * sizeof(DATA_TYPE));
  DATA_TYPE *B = (DATA_TYPE*)malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE));
  DATA_TYPE *B_outputFromGpu = (DATA_TYPE*)malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE));
  DATA_TYPE *C = (DATA_TYPE*)malloc(4 * 4 * sizeof(DATA_TYPE));

  for (int i = 0; i < MAP_SIZE; ++i)
    for (int j = 0; j < MAP_SIZE; ++j)
      A[i * MAP_SIZE + j] = (rand() / (float)RAND_MAX);

  WinogradConv2D_2x2_filter_transformation(C);

  int tile_n = (MAP_SIZE - 2 + 1) / 2;
  int global_offset[2] = {0, 0};
  int local_work_group_size[2] = {DIM_LOCAL_WORK_GROUP_X, DIM_LOCAL_WORK_GROUP_Y};

  size_t cpu_global_size[2];
  size_t gpu_global_size[2];

  bool pass = true;

  double co_time = 0.0;

  for (int cpu_offset = 0; cpu_offset <= 100; cpu_offset++) {
    cpu_global_size[0] = cpu_offset * (size_t)ceil(((float)tile_n) / ((float)DIM_LOCAL_WORK_GROUP_X)) / 100 * DIM_LOCAL_WORK_GROUP_X;
    cpu_global_size[1] = tile_n;

    gpu_global_size[0] = tile_n - cpu_global_size[0];
    gpu_global_size[1] = tile_n;

    global_offset[0] = cpu_global_size[0];
    global_offset[1] = 0;

    #pragma omp parallel for
    for (int i = 0; i < tile_n; i++) {
      for (int j = 0; j < tile_n; j++) {
        WinogradConv2D_2x2_omp(A, B, C, cpu_global_size);
      }
    }

    double co_start = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    #pragma omp parallel for
    for (int i = 0; i < tile_n; i++) {
      for (int j = 0; j < tile_n; j++) {
        WinogradConv2D_2x2(A, B, C);
      }
    }

    co_time += std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() - co_start;

    pass &= compareResults(B, B_outputFromGpu);
  }

  printf("%s\n", pass ? "PASS" : "FAIL");

  free(A);
  free(B);
  free(B_outputFromGpu);
  free(C);

  double end = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
  printf("Co-execution time: %lf s\n", co_time);
  printf("Total time: %lf s\n", end - start);
  printf("Ratio of co-execution time to total time: %.2lf%%\n", 100.0 * co_time / (end - start));

  return 0;
}