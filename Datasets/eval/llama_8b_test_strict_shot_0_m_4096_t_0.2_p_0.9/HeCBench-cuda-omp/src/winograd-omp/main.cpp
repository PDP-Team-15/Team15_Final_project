#include <chrono>
#include <omp.h>
#include "utils.h"
#include <omp.h>
#include <chrono>
#include <cstdlib>
#include <cstring>

void winograd_conv2d(
    const DATA_TYPE *__restrict__ input,
    const DATA_TYPE *__restrict__ transformed_filter ,
    DATA_TYPE *__restrict__ output,
    const int offset_i,
    const int offset_j)
{
  int tile_i = omp_get_thread_num() / omp_get_num_threads() * 2 + offset_i;
  int tile_j = omp_get_thread_num() % omp_get_num_threads() * 2 + offset_j;

  DATA_TYPE input_tile[4][4], tmp_tile[4][4], transformed_tile[4][4];
  #pragma omp parallel for
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

  #pragma omp parallel for
  for (int j = 0; j < 4; j ++) {
    tmp_tile[0][j] = input_tile[0][j] - input_tile[2][j];
    tmp_tile[1][j] = input_tile[1][j] + input_tile[2][j];
    tmp_tile[2][j] = -input_tile[1][j] + input_tile[2][j];
    tmp_tile[3][j] = input_tile[1][j] - input_tile[3][j];
  }
  

  #pragma omp parallel for
  for (int i = 0; i < 4; i ++) {
    transformed_tile[i][0] = tmp_tile[i][0] - tmp_tile[i][2];
    transformed_tile[i][1] = tmp_tile[i][1] + tmp_tile[i][2];
    transformed_tile[i][2] = -tmp_tile[i][1] + tmp_tile[i][2];
    transformed_tile[i][3] = tmp_tile[i][1] - tmp_tile[i][3];
  }

  

  DATA_TYPE multiplied_tile[4][4];
  #pragma omp parallel for
  for (int i = 0; i < 4; i ++) {
    for (int j = 0; j < 4; j ++) {
      multiplied_tile[i][j] = transformed_tile[i][j] * transformed_filter[i * 4 + j];
    }
  }

  

  DATA_TYPE tmp_tile_1[2][4], final_tile[2][2];

  

  #pragma omp parallel for
  for (int j = 0; j < 4; j ++) {
    tmp_tile_1[0][j] = multiplied_tile[0][j] + multiplied_tile[1][j] + multiplied_tile[2][j];
    tmp_tile_1[1][j] = multiplied_tile[1][j] - multiplied_tile[2][j] - multiplied_tile[3][j];
  }
  

  #pragma omp parallel for
  for (int i = 0; i < 2; i ++) {
    final_tile[i][0] = tmp_tile_1[i][0] + tmp_tile_1[i][1] + tmp_tile_1[i][2];
    final_tile[i][1] = tmp_tile_1[i][1] - tmp_tile_1[i][2] - tmp_tile_1[i][3];
  }

  #pragma omp parallel for
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

  double start = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

  DATA_TYPE *A = (DATA_TYPE*)malloc(MAP_SIZE * MAP_SIZE * sizeof(DATA_TYPE));
  DATA_TYPE *B = (DATA_TYPE*)malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE));
  DATA_TYPE *B_outputFromGpu = (DATA_TYPE*)malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE));
  DATA_TYPE *C = (DATA_TYPE*)malloc(4 * 4 * sizeof(DATA_TYPE));

  for (int i = 0; i < MAP_SIZE; ++i)
    for (int j = 0; j < MAP_SIZE; ++j)
      A[i * MAP_SIZE + j] = rand() / (float)RAND_MAX;

  

  WinogradConv2D_2x2_filter_transformation(C);

  DATA_TYPE *d_A;
  //cudaMalloc((void**)&d_A, MAP_SIZE * MAP_SIZE * sizeof(DATA_TYPE));
  //cudaMemcpy(d_A, A, MAP_SIZE * MAP_SIZE * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

  DATA_TYPE *d_B;
  //cudaMalloc((void**)&d_B, (MAP_SIZE-2) * (MAP_SIZE-2) * sizeof(DATA_TYPE));

  DATA_TYPE *d_C;
  //cudaMalloc((void**)&d_C, 16 * sizeof(DATA_TYPE));
  //cudaMemcpy(d_C, C, 16 * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

  const int tile_n = (MAP_SIZE - 2 + 1) / 2;

  

  int cpu_global_size[2];
  int gpu_global_size[2];
  int global_offset[2];

  bool pass = true;

  

  double co_time = 0.0;

  for (int cpu_offset = 0; cpu_offset <= 100; cpu_offset++) {

    cpu_global_size[0] = cpu_offset * (MAP_SIZE - 2) / 100;
    cpu_global_size[1] = MAP_SIZE - 2;

    gpu_global_size[0] = MAP_SIZE - 2 - cpu_global_size[0];
    gpu_global_size[1] = MAP_SIZE - 2;

    global_offset[0] = cpu_global_size[0];
    global_offset[1] = 0;

    #pragma omp parallel for
    for (int i = 0; i < gpu_global_size[0]; i++) {
      for (int j = 0; j < gpu_global_size[1]; j++) {
        winograd_conv2d(A, C, B, i, j);
      }
    }

    double co_start = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    #pragma omp parallel for
    for (int i = 0; i < cpu_global_size[0]; i++) {
      for (int j = 0; j < cpu_global_size[1]; j++) {
        WinogradConv2D_2x2_omp(A, B, C, i, j);
      }
    }

    cudaMemcpy(B_outputFromGpu, B, (MAP_SIZE-2) * (MAP_SIZE-2) * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    co_time += std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() - co_start;

#ifdef VERBOSE
    printf("run on host\n");
    printf("run on device\n");
    printf("CPU workload size : %d\n", cpu_offset);
#endif

    WinogradConv2D_2x2(A, B, C);
    pass &= compareResults(B, B_outputFromGpu);

  } 


  printf("%s\n", pass ? "PASS" : "FAIL");

  //cudaFree(d_A);
  //cudaFree(d_B);
  //cudaFree(d_C);
  free(A);
  free(B);
  free(B_outputFromGpu);
  free(C);

  double end = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
  printf("Co-execution time: %lf s\n", co_time);
  printf("Total time: %lf s\n", end - start);
  printf("Ratio of co-execution time to total time: %.2lf%%\n",
         100.0 * co_time / (end - start));

  return 0;
}