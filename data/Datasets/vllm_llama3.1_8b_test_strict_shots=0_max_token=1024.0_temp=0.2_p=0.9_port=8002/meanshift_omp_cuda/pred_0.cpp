```cpp
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include "utils.h"
#include "constants.h"

namespace mean_shift::gpu {
  __global__ void mean_shift_kernel(float *data, float *data_next, int N, int D, int RADIUS, float DBL_SIGMA_SQ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
      int row = tid * D;
      float new_position[D] = {0.f};
      float tot_weight = 0.f;
      for (int i = 0; i < N; ++i) {
        int row_n = i * D;
        float sq_dist = 0.f;
        for (int j = 0; j < D; ++j) {
          sq_dist += (data[row + j] - data[row_n + j]) * (data[row + j] - data[row_n + j]);
        }
        if (sq_dist <= RADIUS) {
          float weight = expf(-sq_dist / DBL_SIGMA_SQ);
          for (int j = 0; j < D; ++j) {
            new_position[j] += weight * data[row_n + j];
          }
          tot_weight += weight;
        }
      }
      for (int j = 0; j < D; ++j) {
        data_next[row + j] = new_position[j] / tot_weight;
      }
    }
  }

  void mean_shift(const float *data, float *data_next, const int teams, const int threads) {
    int N = mean_shift::gpu::N;
    int D = mean_shift::gpu::D;
    int RADIUS = mean_shift::gpu::RADIUS;
    float DBL_SIGMA_SQ = mean_shift::gpu::DBL_SIGMA_SQ;
    int BLOCKS = teams;
    int TILE_WIDTH = threads;

    float *d_data, *d_data_next;
    cudaMalloc((void **)&d_data, N * D * sizeof(float));
    cudaMalloc((void **)&d_data_next, N * D * sizeof(float));
    cudaMemcpy(d_data, data, N * D * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_WIDTH);
    dim3 gridSize(BLOCKS);
    mean_shift_kernel<<<gridSize, blockSize>>>(d_data, d_data_next, N, D, RADIUS, DBL_SIGMA_SQ);
    cudaDeviceSynchronize();
    cudaMemcpy(data_next, d_data_next, N * D * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_data_next);
  }

  __global__ void mean_shift_tiling_kernel(float *data, float *data_next, int N, int D, int RADIUS, float DBL_SIGMA_SQ, int TILE_WIDTH, int BLOCKS) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    int bid = blockIdx.x;
    int tid_in_tile = bid * TILE_WIDTH + lid;
    int row = tid * D;
    int local_row = lid * D;
    float new_position[D] = {0.f};
    float tot_weight = 0.f;

    for (int t = 0; t < BLOCKS; ++t) {
      if (tid_in_tile < N) {
        int row_in_tile = tid_in_tile * D;
        for (int j = 0; j < D; ++j) {
          data[local_row + j] = data[row_in_tile + j];
        }
      }
      else {
        for (int j = 0; j < D; ++j) {
          data[local_row + j] = 0;
        }
      }
      __syncthreads();
      for (int i = 0; i < TILE_WIDTH; ++i) {
        int local_row_tile = i * D;
        float valid_radius = RADIUS * (tid_in_tile == i ? 1 : 0);
        float sq_dist = 0.;
        for (int j = 0; j < D; ++j) {
          sq_dist += (data[row + j] - data[local_row_tile + j]) * (data[row + j] - data[local_row_tile + j]);
        }
        if (sq_dist <= valid_radius) {
          float weight = expf(-sq_dist / DBL_SIGMA_SQ);
          for (int j = 0; j < D; ++j) {
            new_position[j] += (weight * data[local_row_tile + j]);
          }
          tot_weight += (weight * (tid_in_tile == i ? 1 : 0));
        }
      }
      __syncthreads();
    }
    if (tid < N) {
      for (int j = 