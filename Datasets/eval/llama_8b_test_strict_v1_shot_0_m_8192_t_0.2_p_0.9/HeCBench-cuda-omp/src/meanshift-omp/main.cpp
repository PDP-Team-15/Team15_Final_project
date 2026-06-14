#include <math.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <omp.h>
#include "utils.h"
#include "constants.h"
#include <stdio.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <array>

#define N 1000
#define D 10
#define M 100
#define THREADS 256
#define BLOCKS 100
#define TILE_WIDTH 16
#define DIST_TO_REAL 1e-6
#define NUM_ITER 10
#define MIN_DISTANCE 1e-6
#define RADIUS 1.0
#define DBL_SIGMA_SQ 1.0

namespace mean_shift::gpu {
  void mean_shift(const float *data, float *data_next) {
    int tid = omp_get_thread_num();
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

  void mean_shift_tiling(const float* data, float* data_next) {
    int tid = omp_get_thread_num();
    int row = tid * D;
    int local_row = tid * D;
    float new_position[D] = {0.f};
    float tot_weight = 0.f;

    #pragma omp parallel for num_threads(BLOCKS)
    for (int t = 0; t < BLOCKS; ++t) {
      int tid_in_tile = t * TILE_WIDTH + tid;
      if (tid_in_tile < N) {
        int row_in_tile = tid_in_tile * D;
        for (int j = 0; j < D; ++j) {
          data[row_in_tile + j] = data[row_in_tile + j];
        }
      }
      else {
        for (int j = 0; j < D; ++j) {
          data[row_in_tile + j] = 0;
        }
      }
    }

    #pragma omp parallel for num_threads(BLOCKS)
    for (int t = 0; t < BLOCKS; ++t) {
      int tid_in_tile = t * TILE_WIDTH + tid;
      if (tid_in_tile < N) {
        int row_in_tile = tid_in_tile * D;
        for (int j = 0; j < D; ++j) {
          data[row_in_tile + j] = data[row_in_tile + j];
        }
      }
      else {
        for (int j = 0; j < D; ++j) {
          data[row_in_tile + j] = 0;
        }
      }
    }

    #pragma omp parallel for num_threads(BLOCKS)
    for (int t = 0; t < BLOCKS; ++t) {
      int tid_in_tile = t * TILE_WIDTH + tid;
      if (tid_in_tile < N) {
        int row_in_tile = tid_in_tile * D;
        for (int j = 0; j < D; ++j) {
          data[row_in_tile + j] = data[row_in_tile + j];
        }
      }
      else {
        for (int j = 0; j < D; ++j) {
          data[row_in_tile + j] = 0;
        }
      }
    }

    if (tid < N) {
      for (int j = 0; j < D; ++j) {
        data_next[row + j] = new_position[j] / tot_weight;
      }
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

  mean_shift::gpu::utils::print_info(path_to_data, N, D, BLOCKS, THREADS, TILE_WIDTH);

  std::array<float, M * D> real = mean_shift::gpu::utils::load_csv<M, D>(path_to_centroids, ',');
  std::array<float, N * D> data = mean_shift::gpu::utils::load_csv<N, D>(path_to_data, ',');
  std::array<float, N * D> result;

  for (size_t i = 0; i < mean_shift::gpu::NUM_ITER; ++i) {
    #pragma omp parallel for num_threads(BLOCKS)
    for (int j = 0; j < N; ++j) {
      mean_shift::gpu::mean_shift(data.data(), result.data());
    }
  }

  auto start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < mean_shift::gpu::NUM_ITER; ++i) {
    #pragma omp parallel for num_threads(BLOCKS)
    for (int j = 0; j < N; ++j) {
      mean_shift::gpu::mean_shift_tiling(data.data(), result.data());
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "\nAverage execution time of mean-shift (base) "
            << (time * 1e-6f) / mean_shift::gpu::NUM_ITER << " ms\n" << std::endl;

  std::array<float, N * D> centroids = mean_shift::gpu::utils::reduce_to_centroids<N, D>(result, mean_shift::gpu::MIN_DISTANCE);
  bool are_close = mean_shift::gpu::utils::are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
  if (centroids.size() == M && are_close)
     std::cout << "PASS\n";
  else
     std::cout << "FAIL\n";

  return 0;
}