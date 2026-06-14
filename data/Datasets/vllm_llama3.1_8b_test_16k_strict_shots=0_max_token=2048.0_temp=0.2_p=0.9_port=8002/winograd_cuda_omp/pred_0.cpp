```cpp
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
  int tile_i = omp_get_block_num(0) * omp_get_block_size(0) + omp_get_thread_num(0) + offset_i;
  int tile_j = omp_get_block_num(1) * omp_get_block_size(1) + omp_get_thread_num(1) + offset_j;

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

  double start = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

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

  size_t globalWorkSize[2] = {
    (size_t)ceil(((float)tile_n) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X,
    (size_t)ceil(((float)tile_n) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y };

  size_t localWorkSize[2] = {DIM_LOCAL_WORK_GROUP_X, DIM_LOCAL_WORK_GROUP_Y};

  size_t cpu_global_size[2];
  size_t gpu_global_size[2];
  int global_offset[2];

  bool pass = true;

  double co_time = 0.0;

  for (int cpu_offset = 0; cpu_offset <= 100; cpu_offset++) {

    cpu_global_size[0] = cpu_offset * (size_t)ceil(((float)tile_n) / ((float)DIM_LOCAL_WORK_GROUP_X)) 
      / 100 * DIM_LOCAL_WORK_GROUP_X;
    cpu_global_size[1] = globalWorkSize[1];

    gpu_global_size[0] = globalWorkSize[0] - cpu_global_size[0];
    gpu_global_size[1] = globalWorkSize[1];

    global_offset[0] = cpu_global_size[0];
    global_offset[1] = 0;

    #pragma omp parallel num_threads(gpu_global_size[0] / DIM_LOCAL_WORK_GROUP_X * gpu_global_size[1] / DIM_LOCAL_WORK_GROUP_Y)
    {
      int thread_num = omp_get_thread_num();
      int block_num_x = thread_num / (gpu_global_size[1] / DIM_LOCAL_WORK_GROUP_Y);
      int block_num_y = thread_num % (gpu_global_size[1] / DIM_LOCAL_WORK_GROUP_Y);
      int block_size_x = gpu_global_size[0] / (gpu_global_size[1] / DIM_LOCAL_WORK_GROUP_Y);
      int block_size_y = DIM_LOCAL_WORK_GROUP_Y;
      int block_id_x = block_num_x * block_size_x;
      int block_id_y = block_num_y * block_size_y;
      int thread_id_x = thread_num % (gpu_global_size[0] / (gpu_global_size[1] / DIM_LOCAL_WORK_GROUP_Y));
      int thread_id_y = thread_num % (gpu_global_size[1] / DIM_LOCAL_WORK_GROUP_Y);
      int tile_i = block_id_x + thread_id_x + global_offset[0];
      int tile_j = block_id_y + thread_id_y + global_offset[1];

      DATA_TYPE input_tile[4][4], tmp_tile[4][4], transformed_tile[4][4];
      #pragma omp for
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

      #pragma omp for
      for (int j = 0; j < 4; j ++) {
        tmp_tile[0][j] = input_tile[0][j] - input_tile[2][j];
        tmp_tile[1][j] = input_tile[1][j] + input_tile[2][j];
        tmp_tile[2][j] = -input_tile[1][j] + input_tile[2][j];
        tmp_tile[3][j] = input_tile[1][j]