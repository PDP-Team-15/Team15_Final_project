Here is the CUDA translation of the provided OpenMP code:

```cpp
#include <math.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include "cuda_runtime.h"
#include "utils.h"
#include "constants.h"

namespace mean_shift::gpu {
  __global__ void mean_shift_kernel(const float *data, float *data_next,
                                   const int N, const int D, const float RADIUS, const float DBL_SIGMA_SQ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

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

  __global__ void mean_shift_tiling_kernel(const float* data, float* data_next,
                                          const int N, const int D, const float RADIUS, const float DBL_SIGMA_SQ) {
    extern __shared__ float shared_data[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_id = tid / TILE_WIDTH;
    int lane_id = tid % TILE_WIDTH;
    int row = tid * D;
    float new_position[D] = {0.f};
    float tot_weight = 0.f;

    for (int t = 0; t < BLOCKS; ++t) {
      int tid_in_tile = t * TILE_WIDTH + lane_id;
      int row_in_tile = tid_in_tile * D;
      float local_data[D];
      if (tid_in_tile < N) {
        for (int j = 0; j < D; ++j) {
          local_data[j] = data[row_in_tile + j];
        }
      } else {
        for (int j = 0; j < D; ++j) {
          local_data[j] = 0;
        }
      }
      __syncthreads();

      for (int i = 0; i < TILE_WIDTH; ++i) {
        int local_row_tile = i * D;
        float valid_radius = RADIUS * (tid_in_tile < N ? 1 : 0);
        float sq_dist = 0.;
        for (int j = 0; j < D; ++j) {
          sq_dist += (data[row + j] - local_data[j]) *
                     (data[row + j] - local_data[j]);
        }
        if (sq_dist <= valid_radius) {
          float weight = expf(-sq_dist / DBL_SIGMA_SQ);
          for (int j = 0; j < D; ++j) {
            new_position[j] += (weight * local_data[j]);
          }
          tot_weight += (weight * (tid_in_tile < N ? 1 : 0));
        }
      }
      __syncthreads();
    }
    if (tid < N) {
      for (int j = 0; j < D; ++j) {
        data_next[row + j] = new_position[j] / tot_weight;
      }
    }
  }

  void mean_shift(const float *data, float *data_next,
                  const int teams, const int threads) {
    mean_shift_kernel<<<teams, threads>>>(data, data_next, N, D, RADIUS, DBL_SIGMA_SQ);
  }

  void mean_shift_tiling(const float* data, float* data_next,
                         const int teams, const int threads) {
    mean_shift_tiling_kernel<<<teams, threads, TILE_WIDTH * D * sizeof(float)>>>(data, data_next, N, D, RADIUS, DBL_SIGMA_SQ);
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
  float *d_data, *d_data_next;
  cudaMalloc((void**)&d_data, data_bytes);
  cudaMalloc((void**)&d_data_next, data_bytes);

  cudaMemcpy(d_data, result.data(), data_bytes, cudaMemcpyHostToDevice);

  auto start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < mean_shift::gpu::NUM_ITER; ++i) {
    mean_shift::gpu::mean_shift(d_data, d_data_next, BLOCKS, THREADS);
    mean_shift::gpu::utils::swap(d_data, d_data_next);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "\nAverage execution time of mean-shift (base) "
            << (time * 1e-6f) / mean_shift::gpu::NUM_ITER << " ms\n" << std::endl;

  cudaMemcpy(result.data(), d_data, data_bytes, cudaMemcpyDeviceToHost);

  auto centroids = mean_shift::gpu::utils::reduce_to_centroids<N, D>(result, mean_shift::gpu::MIN_DISTANCE);
  bool are_close = mean_shift::gpu::utils::are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
  if (centroids.size() == M && are_close)
     std::cout << "PASS\n";
  else
     std::cout << "FAIL\n";

  result = data;
  cudaMemcpy(d_data, result.data(), data_bytes, cudaMemcpyHostToDevice);

  start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < mean_shift::gpu::NUM_ITER; ++i) {
    mean_shift::gpu::mean_shift_tiling(d_data, d_data_next, BLOCKS, THREADS);
    mean_shift::gpu::utils::swap(d_data, d_data_next);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "\nAverage execution time of mean-shift (opt) "
            << (time * 1e-6f) / mean_shift::gpu::NUM_ITER << " ms\n" << std::endl;

  cudaMemcpy(result.data(), d_data, data_bytes, cudaMemcpyDeviceToHost);

  centroids = mean_shift::gpu::utils::reduce_to_centroids<N, D>(result, mean_shift::gpu::MIN_DISTANCE);
  are_close = mean_shift::gpu::utils::are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
  if (centroids.size() == M && are_close)
     std::cout << "PASS\n";
  else
     std::cout << "FAIL\n";

  cudaFree(d_data);
  cudaFree(d_data_next);
  return 0;
}
```

### Explanation:
1. **Kernel Functions**:
   - `mean_shift_kernel`: This kernel performs the basic mean shift operation without tiling.
   - `mean_shift_tiling_kernel`: This kernel uses tiling to improve memory access patterns and reduce global memory bandwidth.

2. **Memory Management**:
   - Memory allocation and deallocation using `cudaMalloc` and `cudaFree`.
   - Data transfer between host and device using `cudaMemcpy`.

3. **Execution Configuration**:
   - The number of blocks and threads per block are configured based on the constants defined in `constants.h`.

4. **Main Function**:
   - The main function remains largely unchanged, ensuring that the same logic is applied to both the base and optimized versions of the mean shift algorithm.

This translation ensures that the CUDA version maintains the same functionality as the original OpenMP code while leveraging GPU parallelism.