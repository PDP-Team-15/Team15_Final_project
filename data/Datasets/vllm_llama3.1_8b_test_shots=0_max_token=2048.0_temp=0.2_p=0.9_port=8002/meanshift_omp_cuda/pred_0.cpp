Here's the translation of the provided code from OpenMP to CUDA:

```cpp
#include <math.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include "utils.h"
#include "constants.h"

namespace mean_shift::gpu {
  __global__ void mean_shift_kernel(const float *data, float *data_next,
                                     const int N, const int D, const int RADIUS, const float DBL_SIGMA_SQ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
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

  __global__ void mean_shift_tiling_kernel(const float *data, float *data_next,
                                            const int N, const int D, const int RADIUS, const float DBL_SIGMA_SQ,
                                            const int TILE_WIDTH, const int BLOCKS) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = tid * D;
    int local_row = threadIdx.x * D;
    float new_position[D] = {0.f};
    float tot_weight = 0.f;

    for (int t = 0; t < BLOCKS; ++t) {
      int tid_in_tile = t * TILE_WIDTH + threadIdx.x;
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

      for (int i = threadIdx.x; i < TILE_WIDTH; i += blockDim.x) {
        int local_row_tile = i * D;
        float valid_radius = RADIUS * (i < TILE_WIDTH ? 1 : 0);
        float sq_dist = 0.;
        for (int j = 0; j < D; ++j) {
          sq_dist += (data[row + j] - data[local_row_tile + j]) *
                     (data[row + j] - data[local_row_tile + j]);
        }
        if (sq_dist <= valid_radius) {
          float weight = expf(-sq_dist / DBL_SIGMA_SQ);
          for (int j = 0; j < D; ++j) {
            new_position[j] += (weight * data[local_row_tile + j]);
          }
          tot_weight += (weight * (i < TILE_WIDTH ? 1 : 0));
        }
      }
      __syncthreads();

      if (tid < N) {
        for (int j = 0; j < D; ++j) {
          data_next[row + j] = new_position[j] / tot_weight;
        }
      }
    }
  }

  void mean_shift(const float *data, float *data_next,
                  const int teams, const int threads) {
    int N = mean_shift::gpu::N;
    int D = mean_shift::gpu::D;
    int RADIUS = mean_shift::gpu::RADIUS;
    float DBL_SIGMA_SQ = mean_shift::gpu::DBL_SIGMA_SQ;

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    mean_shift_kernel<<<numBlocks, blockSize>>>(data, data_next, N, D, RADIUS, DBL_SIGMA_SQ);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
  }

  void mean_shift_tiling(const float* data, float* data_next,
                         const int teams, const int threads) {
    int N = mean_shift::gpu::N;
    int D = mean_shift::gpu::D;
    int RADIUS = mean_shift::gpu::RADIUS;
    float DBL_SIGMA_SQ = mean_shift::gpu::DBL_SIGMA_SQ;
    int TILE_WIDTH = mean_shift::gpu::TILE_WIDTH;
    int BLOCKS = mean_shift::gpu::BLOCKS;

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    mean_shift_tiling_kernel<<<numBlocks, blockSize>>>(data, data_next, N, D, RADIUS, DBL_SIGMA_SQ, TILE_WIDTH, BLOCKS);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <path to data> <path to centroids>" << std::endl;
    return 1;
  }
  const auto path_to_data = argv[1];
  const auto path_to_centroids = argv[2];

  constexpr auto N = mean_shift::gpu::N;
  constexpr auto D = mean_shift::gpu::D;
  constexpr auto M = mean_shift::gpu::M;
  constexpr auto THREADS = mean_shift::gpu::THREADS;
  constexpr auto BLOCKS = mean_shift::gpu::BLOCKS;
  constexpr auto TILE_WIDTH = mean_shift::gpu::TILE_WIDTH;
  constexpr auto DIST_TO_REAL = mean_shift::gpu::DIST_TO_REAL;

  mean_shift::gpu::utils::print_info(path_to_data, N, D, BLOCKS, THREADS, TILE_WIDTH);

  const std::array<float, M * D> real = mean_shift::gpu::utils::load_csv<M, D>(path_to_centroids, ',');
  std::array<float, N * D> data = mean_shift::gpu::utils::load_csv<N, D>(path_to_data, ',');
  std::array<float, N * D> result = data;

  size_t data_bytes = N * D * sizeof(float);
  float *d_data = result.data();
  float *d_data_next = (float*) malloc (data_bytes);

  cudaMalloc((void**)&d_data_next, data_bytes);
  cudaMemcpy(d_data, d_data_next, data_bytes, cudaMemcpyDeviceToDevice);

  auto start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < mean_shift::gpu::NUM_ITER; ++i) {
    mean_shift::gpu::mean_shift(d_data, d_data_next);
    mean_shift::gpu::utils::swap(d_data, d_data_next);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "\nAverage execution time of mean-shift (base) "
            << (time * 1e-6f) / mean_shift::gpu::NUM_ITER << " ms\n" << std::endl;

  mean_shift::gpu::utils::reduce_to_centroids<N, D>(result, mean_shift::gpu::MIN_DISTANCE);
  bool are_close = mean_shift::gpu::utils::are_close_to_real<M, D>(result, real, DIST_TO_REAL);
  if (result.size() == M && are_close)
    std::cout << "PASS\n";
  else
    std::cout << "FAIL\n";

  result = data;
  start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < mean_shift::gpu::NUM_ITER; ++i) {
    mean_shift::gpu::mean_shift_tiling(d_data, d_data_next);
    mean_shift::gpu::utils::swap(d_data, d_data_next);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "\nAverage execution time of mean-shift (opt) "
            << (time * 1e-6f) / mean_shift::gpu::NUM_ITER << " ms\n" << std::endl;

  mean_shift::gpu::utils::reduce_to_centroids<N, D>(result, mean_shift::gpu::MIN_DISTANCE);
  are_close = mean_shift::gpu::utils::are_close_to_real<M, D>(result, real, DIST_TO_REAL);
  if (result.size() == M && are_close)
    std::cout << "PASS\n";
  else
    std::cout << "FAIL\n";

  free(d_data_next);
  cuda