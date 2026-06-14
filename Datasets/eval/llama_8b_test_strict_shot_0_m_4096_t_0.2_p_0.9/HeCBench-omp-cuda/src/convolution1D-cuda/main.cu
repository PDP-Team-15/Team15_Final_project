#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
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
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < input_width) {
    T s = 0;
    int start = idx - mask_width / 2;
    for (int j = 0; j < mask_width; j++) {
      if (start + j >= 0 && start + j < input_width) {
        s += in[start + j] * mask[j];
      }
    }
    out[idx] = s;
  }
}

template<typename T>
void conv1d(const T * __restrict__ mask,
            const T * __restrict__ in,
                  T * __restrict__ out,
            const int input_width,
            const int mask_width)
{
  int threads = BLOCK_SIZE;
  int blocks = (input_width + threads - 1) / threads;
  conv1d_kernel<T><<<blocks, threads>>>(mask, in, out, input_width, mask_width);
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
  int threads = BLOCK_SIZE;
  int blocks = (input_width + threads - 1) / threads;
  int teams = (input_width + threads - 1) / threads;
  conv1d_tiled_kernel<T><<<teams, threads>>>(mask, in, out, input_width, mask_width);
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
                          const T * __restrict__ in,
                                T * __restrict__ out,
                          const int input_width,
                          const int mask_width)
{
  int threads = BLOCK_SIZE;
  int blocks = (input_width + threads - 1) / threads;
  conv1d_tiled_caching_kernel<T><<<blocks, threads>>>(mask, in, out, input_width, mask_width);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in conv1d_tiled_caching kernel: %s\n", cudaGetErrorString(err));
  }
}

template <typename T>
void reference(const T *h_in,
               const T *d_out,
               const T *mask,
               const int input_width,
               const int mask_width)
{
  bool ok = true;
  for (int i = 0; i < input_width; i++) {
    T s = 0;
    int start = i - mask_width / 2;
    for (int j = 0; j < mask_width; j++) {
      if (start + j >= 0 && start + j < input_width) {
        s += h_in[start + j] * mask[j];
      }
    }
    if (fabs(s - d_out[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
}

template <typename T>
void conv1D(const int input_width, const int mask_width, const int repeat)
{
  size_t size_bytes = input_width * sizeof(T);

  T *a, *b;
  a = (T *)malloc(size_bytes); 

  b = (T *)malloc(size_bytes); 


  T mask[MAX_MASK_WIDTH];

  for (int i = 0; i < MAX_MASK_WIDTH; i++) mask[i] = 1; 

  srand(123);
  for (int i = 0; i < input_width; i++) {
    a[i] = rand() % 256;
  }

  T *d_a, *d_b, *d_mask;
  cudaMalloc((void **)&d_a, size_bytes);
  cudaMalloc((void **)&d_b, size_bytes);
  cudaMalloc((void **)&d_mask, MAX_MASK_WIDTH * sizeof(T));
  cudaMemcpy(d_mask, mask, MAX_MASK_WIDTH * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, a, size_bytes, cudaMemcpyHostToDevice);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    conv1d(d_mask, d_a, d_b, input_width, mask_width);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv1d kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  T *h_b;
  h_b = (T *)malloc(size_bytes);
  cudaMemcpy(h_b, d_b, size_bytes, cudaMemcpyDeviceToHost);
  reference(a, h_b, d_mask, input_width, mask_width);
  free(h_b);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    conv1d_tiled(d_mask, d_a, d_b, input_width, mask_width);
  }
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv1d-tiled kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  cudaMemcpy(h_b, d_b, size_bytes, cudaMemcpyDeviceToHost);
  reference(a, h_b, d_mask, input_width, mask_width);
  free(h_b);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    conv1d_tiled_caching(d_mask, d_a, d_b, input_width, mask_width);
  }
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv1d-tiled-caching kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  cudaMemcpy(h_b, d_b, size_bytes, cudaMemcpyDeviceToHost);
  reference(a, h_b, d_mask, input_width, mask_width);
  free(h_b);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_mask);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <input_width> <repeat>\n", argv[0]);
    return 1;
  }

  int input_width = atoi(argv[1]);
  

  input_width = (input_width + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

  const int repeat = atoi(argv[2]);

  for (int mask_width = 3; mask_width < MAX_MASK_WIDTH; mask_width += 2) {
    printf("\n---------------------\n");
    printf("Mask width: %d\n", mask_width); 

    printf("1D convolution (FP64)\n");
    conv1D<double>(input_width, mask_width, repeat);

    printf("1D convolution (FP32)\n");
    conv1D<float>(input_width, mask_width, repeat);

    printf("1D convolution (INT16)\n");
    conv1D<int16_t>(input_width, mask_width, repeat);
  }

  return 0;
}