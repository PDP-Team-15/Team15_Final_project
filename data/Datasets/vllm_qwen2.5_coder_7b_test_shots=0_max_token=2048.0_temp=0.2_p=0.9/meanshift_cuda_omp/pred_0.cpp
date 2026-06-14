Here is the translation of the CUDA code to OpenMP:

```cpp
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <omp.h>
#include "utils.h"
#include "constants.h"

namespace mean_shift::gpu {
  void mean_shift(const float *data, float *data_next) {
    #pragma omp parallel for reduction(+:new_position[:D], tot_weight)
    for (size_t tid = 0; tid < N; ++tid) {
      size_t row = tid * D;
      float new_position[D] = {0.f};
      float tot_weight = 0.f;
      for (size_t i = 0; i < N; ++i) {
        size_t row_n = i * D;
        float sq_dist = 0.f;
        for (size_t j = 0; j < D; ++j) {
          sq_dist += (data[row + j] - data[row_n + j]) * (data[row + j] - data[row_n + j]);
        }
        if (sq_dist <= RADIUS) {
          float weight = expf(-sq_dist / DBL_SIGMA_SQ);
          for (size_t j = 0; j < D; ++j) {
            new_position[j] += weight * data[row_n + j];
          }
          tot_weight += weight;
        }
      }
      for (size_t j = 0; j < D; ++j) {
        data_next[row + j] = new_position[j] / tot_weight;
      }
    }
  }

  void mean_shift_tiling(const float* data, float* data_next) {
    #pragma omp parallel for collapse(2) reduction(+:new_position[:D], tot_weight)
    for (int t = 0; t < BLOCKS; ++t) {
      for (int tid_in_tile = 0; tid_in_tile < TILE_WIDTH; ++tid_in_tile) {
        int tid = t * TILE_WIDTH + tid_in_tile;
        int row = tid * D;
        int local_row = tid_in_tile * D;
        float new_position[D] = {0.f};
        float tot_weight = 0.f;

        if (tid < N) {
          int row_in_tile = tid * D;
          for (int j = 0; j < D; ++j) {
            local_data[local_row + j] = data[row_in_tile + j];
          }
          valid_data[tid_in_tile] = 1;
        } else {
          for (int j = 0; j < D; ++j) {
            local_data[local_row + j] = 0;
          }
          valid_data[tid_in_tile] = 0;
        }

        for (int i = 0; i < TILE_WIDTH; ++i) {
          int local_row_tile = i * D;
          float valid_radius = RADIUS * valid_data[i];
          float sq_dist = 0.;
          for (int j = 0; j < D; ++j) {
            sq_dist += (data[row + j] - local_data[local_row_tile + j]) *
                       (data[row + j] - local_data[local_row_tile + j]);
          }
          if (sq_dist <= valid_radius) {
            float weight = expf(-sq_dist / DBL_SIGMA_SQ);
            for (int j = 0; j < D; ++j) {
              new_position[j] += (weight * local_data[local_row_tile + j]);
            }
            tot_weight += (weight * valid_data[i]);
          }
        }
      }
    }

    #pragma omp parallel for
    for (size_t tid = 0; tid < N; ++tid) {
      size_t row = tid * D;
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

  float *d_data;
  float *d_data_next;
  size_t data_bytes = N * D * sizeof(float);
  // Simulating memory allocation and copying for demonstration purposes
  d_data = new float[data_bytes];
  d_data_next = new float[data_bytes];
  memcpy(d_data, data.data(), data_bytes);

  auto start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < mean_shift::gpu::NUM_ITER; ++i) {
    mean_shift::gpu::mean_shift(d_data, d_data_next);
    mean_shift::gpu::utils::swap(d_data, d_data_next);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "\nAverage execution time of mean-shift (base) "
            << (time * 1e-6f) / mean_shift::gpu::NUM_ITER << " ms\n" << std::endl;

  memcpy(result.data(), d_data, data_bytes);
  auto centroids = mean_shift::gpu::utils::reduce_to_centroids<N, D>(result, mean_shift::gpu::MIN_DISTANCE);
  bool are_close = mean_shift::gpu::utils::are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
  if (centroids.size() == M && are_close)
     std::cout << "PASS\n";
  else
     std::cout << "FAIL\n";

  memcpy(d_data, data.data(), data_bytes);

  start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < mean_shift::gpu::NUM_ITER; ++i) {
    mean_shift::gpu::mean_shift_tiling(d_data, d_data_next);
    mean_shift::gpu::utils::swap(d_data, d_data_next);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "\nAverage execution time of mean-shift (opt) "
            << (time * 1e-6f) / mean_shift::gpu::NUM_ITER << " ms\n" << std::endl;

  memcpy(result.data(), d_data, data_bytes);

  centroids = mean_shift::gpu::utils::reduce_to_centroids<N, D>(result, mean_shift::gpu::MIN_DISTANCE);
  are_close = mean_shift::gpu::utils::are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
  if (centroids.size() == M && are_close)
     std::cout << "PASS\n";
  else
     std::cout << "FAIL\n";

  delete[] d_data;
  delete[] d_data_next;
  return 0;
}
```

### Explanation:
1. **Parallel For Loops**: The `#pragma omp parallel for` directive is used to parallelize the loops over threads.
2. **Reduction Clauses**: The `reduction(+:new_position[:D], tot_weight)` clause ensures that the `new_position` array and `tot_weight` variable are correctly reduced across all threads.
3. **Collapse Clause**: The `collapse(2)` clause is used to collapse nested loops into a single loop iteration, which can improve performance on shared memory architectures.
4. **Memory Allocation and Copying**: Since OpenMP does not handle device memory like CUDA, we simulate memory allocation and copying using standard C++ arrays.

This translation should provide a similar performance profile as the original CUDA code when run on a multi-core CPU with OpenMP.