```cpp
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <omp.h>
#include "utils.h"
#include "constants.h"

namespace mean_shift::gpu {
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
        float valid_radius = RADIUS * (i == tid ? 1 : 0);
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
          tot_weight += (weight * (i == tid ? 1 : 0));
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
  std::array<float, N * D> result;

  float *data_host = new float[N * D];
  float *data_next_host = new float[N * D];
  float *data_device;
  float *data_next_device;
  size_t data_bytes = N * D * sizeof(float);
  data_host = new float[N * D];
  data_next_host = new float[N * D];

  #pragma omp parallel for
  for (int i = 0; i < N * D; ++i) {
    data_host[i] = data[i];
  }

  data_device = new float[N * D];
  data_next_device = new float[N * D];

  #pragma omp parallel for
  for (int i = 0; i < N * D; ++i) {
    data_device[i] = data_host[i];
  }

  auto start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < mean_shift::gpu::NUM_ITER; ++i) {
    #pragma omp parallel for
    for (int j = 0; j < N; ++j) {
      mean_shift(data_device + j * D, data_next_device + j * D);
    }
    #pragma omp parallel for
    for (int j = 0; j < N; ++j) {
      mean_shift::gpu::utils::swap(data_device + j * D, data_next_device + j * D);
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "\nAverage execution time of mean-shift (base) "
            << (time * 1e-6f) / mean_shift::gpu::NUM_ITER << " ms\n" << std::endl;

  #pragma omp parallel for
  for (int i = 0; i < N * D; ++i) {
    result[i] = data_device[i];
  }

  auto centroids = mean_shift::gpu::utils::reduce_to_centroids<N, D>(result, mean_shift::gpu::MIN_DISTANCE);
  bool are_close = mean_shift::gpu::utils::are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
  if (centroids.size() == M && are_close)
     std::cout << "PASS\n";
  else
     std::cout << "FAIL\n";

  #pragma omp parallel for
  for (int i = 0; i < N * D; ++i) {
    data_host[i] = data_device[i];
  }

  start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < mean_shift::gpu::NUM_ITER; ++i) {
    #pragma omp parallel for
    for (int j = 0; j < N; ++j) {
      mean_shift_tiling(data_device + j * D, data_next_device + j * D);
    }
    #pragma omp parallel for
    for (int j = 0; j < N; ++j) {
      mean_shift::gpu::utils::swap(data_device + j * D, data_next_device + j * D);
    }
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "\nAverage execution time of mean-shift (opt) "
            << (time * 1e-6f) / mean_shift::gpu::NUM_ITER << " ms\n" << std::endl;

  #pragma omp parallel for
  for (int i = 0; i < N * D; ++i) {
    result[i] = data_device[i];
  }

  centroids = mean_shift::gpu::utils::reduce_to_centroids<N, D>(result, mean_shift::gpu::MIN_DISTANCE);
  are_close = mean_shift::gpu::utils::are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
  if (centroids.size() == M && are_close)
     std::cout << "PASS\n";
  else
     std::cout << "FAIL\n";

  delete[] data_host;
  delete[] data_next_host;
  delete[] data_device;
  delete[] data_next_device;
  return 0;
}
```
Note that I've used OpenMP's `parallel for` directive to parallelize the loops, and `omp_get_thread_num()` to get the thread ID. I've also used `omp parallel for` to parallelize the loops in the `mean_shift_tiling` function. Additionally, I've used `omp barrier` to ensure that the threads wait for each other to finish before proceeding.