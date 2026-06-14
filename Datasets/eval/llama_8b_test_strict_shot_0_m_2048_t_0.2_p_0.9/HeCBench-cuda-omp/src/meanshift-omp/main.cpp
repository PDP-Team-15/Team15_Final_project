#include <math.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <omp.h>
#include "utils.h"
#include "constants.h"
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <omp.h>

namespace mean_shift::cpu {
  void mean_shift(const float *data, float *data_next) {
    int tid = omp_get_thread_num();
    if (tid < N) {
      int row = tid * D;
      float new_position[D] = {0.f};
      float tot_weight = 0.f;
      #pragma omp parallel for
      for (int i = 0; i < N; ++i) {
        int row_n = i * D;
        float sq_dist = 0.f;
        #pragma omp parallel for reduction(+:sq_dist)
        for (int j = 0; j < D; ++j) {
          sq_dist += (data[row + j] - data[row_n + j]) * (data[row + j] - data[row_n + j]);
        }
        if (sq_dist <= RADIUS) {
          float weight = expf(-sq_dist / DBL_SIGMA_SQ);
          #pragma omp parallel for
          for (int j = 0; j < D; ++j) {
            new_position[j] += weight * data[row_n + j];
          }
          tot_weight += weight;
        }
      }
      #pragma omp parallel for
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

    #pragma omp parallel for
    for (int t = 0; t < BLOCKS; ++t) {
      int tid_in_tile = t * TILE_WIDTH + tid;
      if (tid_in_tile < N) {
        int row_in_tile = tid_in_tile * D;
        #pragma omp parallel for
        for (int j = 0; j < D; ++j) {
          data[row + j] = data[row_in_tile + j];
        }
      }
      else {
        #pragma omp parallel for
        for (int j = 0; j < D; ++j) {
          data[row + j] = 0;
        }
      }
      #pragma omp barrier
      #pragma omp parallel for
      for (int i = 0; i < TILE_WIDTH; ++i) {
        int local_row_tile = i * D;
        float valid_radius = RADIUS * (tid_in_tile < N ? 1 : 0);
        float sq_dist = 0.;
        #pragma omp parallel for reduction(+:sq_dist)
        for (int j = 0; j < D; ++j) {
          sq_dist += (data[row + j] - data[local_row_tile + j]) *
                     (data[row + j] - data[local_row_tile + j]);
        }
        if (sq_dist <= valid_radius) {
          float weight = expf(-sq_dist / DBL_SIGMA_SQ);
          #pragma omp parallel for
          for (int j = 0; j < D; ++j) {
            new_position[j] += (weight * data[local_row_tile + j]);
          }
          tot_weight += (weight * (tid_in_tile < N ? 1 : 0));
        }
      }
      #pragma omp barrier
      if (tid < N) {
        #pragma omp parallel for
        for (int j = 0; j < D; ++j) {
          data_next[row + j] = new_position[j] / tot_weight;
        }
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

  constexpr auto N = mean_shift::cpu::N;
  constexpr auto D = mean_shift::cpu::D;
  constexpr auto M = mean_shift::cpu::M;
  constexpr auto THREADS = mean_shift::cpu::THREADS;
  constexpr auto BLOCKS = mean_shift::cpu::BLOCKS;
  constexpr auto TILE_WIDTH = mean_shift::cpu::TILE_WIDTH;
  constexpr auto DIST_TO_REAL = mean_shift::cpu::DIST_TO_REAL;

  mean_shift::cpu::utils::print_info(path_to_data, N, D, BLOCKS, THREADS, TILE_WIDTH);

  const std::array<float, M * D> real = mean_shift::cpu::utils::load_csv<M, D>(path_to_centroids, ',');
  std::array<float, N * D> data = mean_shift::cpu::utils::load_csv<N, D>(path_to_data, ',');
  std::array<float, N * D> result;

  for (size_t i = 0; i < mean_shift::cpu::NUM_ITER; ++i) {
    mean_shift::cpu::mean_shift(data.data(), result.data());
    mean_shift::cpu::utils::swap(data.data(), result.data());
  }

  auto start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < mean_shift::cpu::NUM_ITER; ++i) {
    mean_shift::cpu::mean_shift_tiling(data.data(), result.data());
    mean_shift::cpu::utils::swap(data.data(), result.data());
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "\nAverage execution time of mean-shift (base) "
            << (time * 1e-6f) / mean_shift::cpu::NUM_ITER << " ms\n" << std::endl;

  auto centroids = mean_shift::cpu::utils::reduce_to_centroids<N, D>(result, mean_shift::cpu::MIN_DISTANCE);
  bool are_close = mean_shift::cpu::utils::are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
  if (centroids.size() == M && are_close)
     std::cout << "PASS\n";
  else
     std::cout << "FAIL\n";

  for (size_t i = 0; i < mean_shift::cpu::NUM_ITER; ++i) {
    mean_shift::cpu::mean_shift_tiling(data.data(), result.data());
    mean_shift::cpu::utils::swap(data.data(), result.data());
  }

  centroids = mean_shift::cpu::utils::reduce_to_centroids<N, D>(result, mean_shift::cpu::MIN_DISTANCE);
  are_close = mean_shift::cpu::utils::are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
  if (centroids.size() == M && are_close)
     std::cout << "PASS\n";
  else
     std::cout << "FAIL\n";

  return 0;
}