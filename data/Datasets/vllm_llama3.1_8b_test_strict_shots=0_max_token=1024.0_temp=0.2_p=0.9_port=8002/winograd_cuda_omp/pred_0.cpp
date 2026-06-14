```cpp
#include <chrono>
#include <omp.h>
#include "utils.h"

void winograd_conv2d(
    const DATA_TYPE *__restrict__ input,
    const DATA_TYPE *__restrict__ transformed_filter ,
    DATA_TYPE *__restrict__ output,
    const int offset_i,
    const int offset_j)
{
  int tile_i = omp_get_block_num(0) * omp_get_block_size(0) + omp_get_thread_num(0) + offset_i;
  int tile_j = omp_get_block_num(1) * omp_get_block_size(1) + omp_get_thread_num(1) + offset_j;

  DATA_TYPE input_tile[4][4], tmp_tile[4][4], transformed_tile[4][4];
  for (int i = 0; i < 4; i ++) {
    for (int j = 0; j < 4; j ++) { 
      int x = 2 * tile_i + i;
      int y = 2 * tile_j + j;
      if (x >= MAP_SIZE || y >= MAP_SIZE) {
        input_tile[i][j] = 0;
        continue;
      }
      input_tile[i][j] = input[x * MAP_SIZE + y];
    }
  } 

  for (int j = 0; j < 4; j ++) {
    tmp_tile[0][j] = input_tile[0][j] - input_tile[2][j];
    tmp_tile[1][j] = input_tile[1][j] + input_tile[2][j];
    tmp_tile[2][j] = -input_tile[1][j] + input_tile[2][j];
    tmp_tile[3][j] = input_tile[1][j] - input_tile[3][j];
  }
  

  for (int i = 0; i < 4; i ++) {
    transformed_tile[i][0] = tmp_tile[i][0] - tmp_tile[i][2];
    transformed_tile[i][1] = tmp_tile[i][1] + tmp_tile[i][2];
    transformed_tile[i][2] = -tmp_tile[i][1] + tmp_tile[i][2];
    transformed_tile[i][3] = tmp_tile[i][1] - tmp_tile[i][3];
  }

  

  DATA_TYPE multiplied_tile[4][4];
  for (int i = 0; i < 4; i ++) {
    for (int j = 0; j < 4; j ++) {
      multiplied_tile[i][j] = transformed_tile[i][j] * transformed_filter[i * 4 + j];
    }
  }

  

  DATA_TYPE tmp_tile_1[2][4], final_tile[2][2];

  

  for (int j = 0; j < 4; j ++) {
    tmp_tile_1[0][j] = multiplied_tile[0][j] + multiplied_tile[1][j] + multiplied_tile[2][j];
    tmp_tile_1[1][j] = multiplied_tile[1][j] - multiplied_tile[2][j] - multiplied_tile[3][j];
  }
  

  for (int i = 0; i < 2; i ++) {
    final_tile[i][0] = tmp_tile_1[i][0] + tmp_tile_1[i][1] + tmp_tile_1[i][2];
    final_tile[i][1] = tmp_tile_1[i][1] - tmp_tile_1[i][2] - tmp_tile_1[i][3];
  }

  for (int i = 0; i < 2; i ++) {
    for (int j = 0; j < 2; j ++) {
      int x = 2 * tile_i + i;
      int y = 2 * tile_j + j;
      if (x >= MAP_SIZE - 2 || y >= MAP_SIZE - 2) {
        continue;
      }
      output[x * (MAP_SIZE - 2) + y] = final_tile[i][j];
    }
  }
}

int main(int argc, char* argv[]) {

  double start = rtclock();

  DATA_TYPE *A = (DATA_TYPE*)malloc(MAP_SIZE * MAP_SIZE * sizeof(DATA_TYPE));
  DATA_TYPE *B = (DATA_TYPE*)malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE));
  DATA_TYPE *B_outputFromGpu = (DATA_TYPE*)malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE));
  DATA_TYPE *C = (DATA_TYPE*)malloc(4 * 4 * sizeof(DATA_TYPE));

  for (int i = 0; i < MAP_SIZE; ++i)
    for (int j = 0; j < MAP_SIZE; ++j)
      A[i