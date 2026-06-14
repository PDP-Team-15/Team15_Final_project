```cpp
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>

#define MAX_MASK_WIDTH 10
#define BLOCK_SIZE 256
#define TILE_SIZE BLOCK_SIZE

template<typename T>
__global__ void conv1d_kernel(const T * __restrict__ mask,
                              const T * __restrict__ in,
                                    T * __restrict__ out,
                              const int input_width,
                              const int mask_width)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < input_width) {
    T s = 0;
    int start = i - mask_width / 2;
    for (int j = 0; j < mask_width; j++) {
      if (start + j >= 0 && start + j < input_width) {
        s += in[start + j] * mask[j];
      }
    }
    out[i] = s;
  }
}

template<typename T>
void conv1d(const T * __restrict__ mask,
            const T * __restrict__ in,
                  T * __restrict__ out,
            const int input_width,
            const int mask_width)
{
  int blockSize = BLOCK_SIZE;
  int numBlocks = (input_width + blockSize - 1) / blockSize;
  conv1d_kernel<T><<<numBlocks, blockSize>>>(mask, in, out, input_width, mask_width);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in conv1d kernel: %s\n", cudaGetErrorString(err));
  }
}

template<typename T>
__global__ void conv1d_tiled_kernel(const T * __restrict__ mask,
                                    const T * __restrict__ in,
                                          T * __restrict__ out,
                                    const int input_width,
                                    const int mask_width)
{
  int bid = blockIdx.x;
  int lid = threadIdx.x;
  int dim = blockDim.x;
  int i = bid * dim + lid;

  int n = mask_width / 2;

  int halo_left = (bid - 1) * dim + lid;
  if (lid >= dim - n)
    tile[lid - (dim - n)] = halo_left < 0 ? 0 : in[halo_left];

  tile[n + lid] = in[bid * dim + lid];

  int halo_right = (bid + 1) * dim + lid;
  if (lid < n)
    tile[lid + dim + n] = halo_right >= input_width ? 0 : in[halo_right];

  __syncthreads();

  T s = 0;
  for (int j = 0; j < mask_width; j++)
    s += tile[lid + j] * mask[j];

  out[i] = s;
}

template<typename T>
void conv1d_tiled(const T * __restrict__ mask,
                  const T * __restrict__ in,
                        T * __restrict__ out,
                  const int input_width,
                  const int mask_width)
{
  int blockSize = BLOCK_SIZE;
  int numTeams = (input_width + blockSize - 1) / blockSize;
  int numBlocks = numTeams;
  conv1d_tiled_kernel<T><<<numBlocks, blockSize>>>(mask, in, out, input_width, mask_width);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in conv1d_tiled kernel: %s\n", cudaGetErrorString(err));
  }
}

template<typename T>
__global__ void conv1d_tiled_caching_kernel(const T * __restrict__ mask,
                                            const T * __restrict__ in,
                                                  T * __restrict__ out,
                                            const int input_width,
                                            const int mask_width)
{
  int bid = blockIdx.x;
  int lid = threadIdx.x;
  int dim = blockDim.x;
  int i = bid * dim + lid;
  tile[lid] = in[i];
  __syncthreads();

  int this_tile_start = bid * dim;
  int next_tile_start = (bid + 1) * dim;
  int start = i - (mask_width / 2);
  T s = 0;
  for (int j = 0; j < mask_width; j++) {
    int in_index = start + j;
    if (in_index >= 0 && in_index < input_width) {
      if (in_index >= this_tile_start && in_index < next_tile_start) {
        s += tile[lid + j - (mask_width / 2)] * mask[j];
      } else {
        s += in[in_index] * mask[j];
      }
    }
  }
  out[i] = s;
}

template<typename T>
void conv1d_tiled_caching(const T * __restrict__ mask,
                         