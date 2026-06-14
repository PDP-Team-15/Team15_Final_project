#include <chrono>
#include <omp.h>
#include "utils.h"
#include <cuda_runtime.h>
#include <chrono>
#include "utils.h"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__global__ void WinogradConv2D_2x2_kernel(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, int tile_i_size, int tile_j_size, int offset_i, int offset_j) {
  int thread_idx_x = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
  int thread_idx_y = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;

  if (thread_idx_x < 4 && thread_idx_y < 4) {
    DATA_TYPE input_tile[4][4], tmp_tile[4][4], transformed_tile[4][4];
    for (int i = 0; i < 4; i ++) {
      for (int j = 0; j < 4; j ++) { 
        int x = 2 * (thread_idx_x + offset_i) + i;
        int y = 2 * (thread_idx_y + offset_j) + j;
        if (x >= MAP_SIZE || y >= MAP_SIZE) {
          input_tile[i][j] = 0;
          continue;
        }
        input_tile[i][j] = A[x * MAP_SIZE + y];
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
        multiplied_tile[i][j] = transformed_tile[i][j] * C[i * 4 + j];
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
        int x = 2 * (thread_idx_x + offset_i) + i;
        int y = 2 * (thread_idx_y + offset_j) + j;
        if (x >= MAP_SIZE - 2 || y >= MAP_SIZE - 2) {
          continue;
        }
        B[x * (MAP_SIZE - 2) + y] = final_tile[i][j];
      }
    }
  }
}

int main(int argc, char* argv[]) {

  double start = rtclock();

  DATA_TYPE *A = (DATA_TYPE*)malloc(MAP_SIZE * MAP_SIZE * sizeof(DATA_TYPE));
  DATA_TYPE *B_host = (DATA_TYPE*)malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE));
  DATA_TYPE *B = (DATA_TYPE*)malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE));
  DATA_TYPE *C = (DATA_TYPE*)malloc(4 * 4 * sizeof(DATA_TYPE));

  for (int i = 0; i < MAP_SIZE; ++i)
    for (int j = 0; j < MAP_SIZE; ++j)
      A[i * MAP_SIZE + j] = rand() / (float)RAND_MAX;

  WinogradConv2D_2x2_filter_transformation(C);

  const int tile_n = (MAP_SIZE - 2 + 1) / 2;

  size_t globalWorkSize[2] = {
    (size_t)ceil(((float)tile_n) / ((float)BLOCK_SIZE_X)) * BLOCK_SIZE_X,
    (size_t)ceil(((float)tile_n) / ((float)BLOCK_SIZE_Y)) * BLOCK_SIZE_Y };

  size_t localWorkSize[2] = {BLOCK_SIZE_X, BLOCK_SIZE_Y};

  size_t cpu_global_size[2];
  size_t gpu_global_size[2];
  size_t global_offset[2];

  bool pass = true;

  double co_time = 0.0;

  for (int cpu_offset = 0; cpu_offset <= 100; cpu_offset++) {

    cpu_global_size[0] = cpu_offset * (size_t)ceil(((float)tile_n) / ((float)BLOCK_SIZE_X)) 
      / 100 * BLOCK_SIZE_X;
    cpu_global_size[1] = globalWorkSize[1];

    gpu_global_size[0] = globalWorkSize[0] - cpu_global_size[0];
    gpu_global_size[1] = globalWorkSize[1];

    global_offset[0] = cpu_global_size[0];
    global_offset[1] = 0;

    const int tile_i_size = gpu_global_size[0];
    const int tile_j_size = gpu_global_size[1];
    const int offset_i = global_offset[0];
    const int offset_j = global_offset[1];
    const int thread_size = localWorkSize[1] * localWorkSize[0];

    bool cpu_run = false, gpu_run = false;
    if (cpu_global_size[0] > 0) {
      cpu_run = true;
    }
    if (gpu_global_size[0] > 0) {
      gpu_run = true;
    }

    double co_start = rtclock();

    if (gpu_run) {
      dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
      dim3 gridSize((tile_i_size + blockSize.x - 1) / blockSize.x, (tile_j_size + blockSize.y - 1) / blockSize.y);
      WinogradConv2D_2x2_kernel<<<gridSize, blockSize>>>(A, B, C, tile_i_size, tile_j_size, offset_i, offset_j);
      cudaDeviceSynchronize();
    }

    if (cpu_run) {
      WinogradConv2D_2x2_omp(A, B, C, cpu_global_size);

      if (gpu_run) {
        cudaMemcpy(B_host, B, (cpu_global_size[0] * (MAP_SIZE - 2) * sizeof(DATA_TYPE)), cudaMemcpyDeviceToHost);
      }
      else {
        cudaMemcpy(B_host, B, (MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
      }
    }

    if (gpu_run) {
      cudaMemcpy(B_host, B, (cpu_global_size[0] * (MAP_SIZE - 2) * sizeof(DATA_TYPE)), cudaMemcpyDeviceToHost);
    }

    co_time += rtclock() - co_start;

#ifdef VERBOSE
    if (cpu_run) printf("run on host\n");
    if (gpu_run) printf("run on device\n");
    printf("CPU workload size : %d\n", cpu_offset);
#endif

    WinogradConv2D_2x2(A, B_host, C);
    pass &= compareResults(B_host, B);

  } 

  printf("%s\n", pass ? "PASS" : "FAIL");

  free(A);
  free(B);
  free(B_host);
  free(C);

  double end = rtclock();
  printf("Co-execution time: %lf s\n", co_time);
  printf("Total time: %lf s\n", end - start);
  printf("Ratio of co-execution time to total time: %.2lf%%\n",
         100.0 * co_time / (end - start));

  return 0;
}