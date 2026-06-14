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
#include <omp.h>
#include <chrono>

#define MAX_MASK_WIDTH 10
#define BLOCK_SIZE 256
#define TILE_SIZE BLOCK_SIZE

template<typename T>
static void conv1d(const T * __restrict__ in,
                   T * __restrict__ out,
                   const int input_width,
                   const int mask_width)
{
  #pragma omp parallel for reduction(+:out[:input_width])
  for (int i = 0; i < input_width; i++) {
    T s = 0;
    int start = i - mask_width / 2;
    for (int j = 0; j < mask_width; j++) {
      if (start + j >= 0 && start + j < input_width) {
        s += in[start + j] * mask<T>[j];
      }
    }
    out[i] = s;
  }
}

template<typename T>
static void conv1d_tiled(const T *__restrict__ in,
                         T *__restrict__ out,
                         const int input_width,
                         const int mask_width)
{
  #pragma omp parallel for collapse(2) reduction(+:out[:input_width])
  for (int blockIdx_x = 0; blockIdx_x < input_width / BLOCK_SIZE; ++blockIdx_x) {
    for (int threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; ++threadIdx_x) {
      __shared__ T tile[TILE_SIZE + MAX_MASK_WIDTH - 1];
      int i = threadIdx_x + blockIdx_x * blockDim.x;

      int n = mask_width / 2;  
      int halo_left = (blockIdx_x - 1) * blockDim.x + threadIdx_x;
      if (threadIdx_x >= blockDim.x - n)
         tile[threadIdx_x - (blockDim.x - n)] = halo_left < 0 ? 0 : in[halo_left];

      tile[n + threadIdx_x] = in[blockIdx_x * blockDim.x + threadIdx_x];

      int halo_right = (blockIdx_x + 1) * blockDim.x + threadIdx_x;
      if (threadIdx_x < n)
         tile[threadIdx_x + blockDim.x + n] = halo_right >= input_width ? 0 : in[halo_right];

      T s = 0;
      for (int j = 0; j < mask_width; j++)
        s += tile[threadIdx_x + j] * mask<T>[j];

      out[i] = s;
    }
  }
}

template<typename T>
static void conv1d_tiled_caching(const T *__restrict__ in,
                                 T *__restrict__ out,
                                 const int input_width,
                                 const int mask_width)
{
  #pragma omp parallel for collapse(2) reduction(+:out[:input_width])
  for (int blockIdx_x = 0; blockIdx_x < input_width / BLOCK_SIZE; ++blockIdx_x) {
    for (int threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; ++threadIdx_x) {
      __shared__ T tile[TILE_SIZE];

      int i = threadIdx_x + blockIdx_x * blockDim.x;
      tile[threadIdx_x] = in[i];
      #pragma omp barrier

      int this_tile_start = blockIdx_x * blockDim.x;
      int next_tile_start = (blockIdx_x + 1) * blockDim.x;
      int start = i - (mask_width / 2);
      T s = 0;
      for (int j = 0; j < mask_width; j++) {
        int in_index = start + j;
        if (in_index >= 0 && in_index < input_width) {
          if (in_index >= this_tile_start && in_index < next_tile_start) {
            s += tile[threadIdx_x + j - (mask_width / 2)] * mask<T>[j];
          } else {
            s += in[in_index] * mask<T>[j];
          }
        }
      }
      out[i] = s;
    }
  }
}

template <typename T>
static void reference(const T *h_in,
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
static void conv1D(const int input_width, const int mask_width, const int repeat)
{
  size_t size_bytes = input_width * sizeof(T);

  T *a, *b;
  a = (T *)malloc(size_bytes); 
  b = (T *)malloc(size_bytes); 

  T h_mask[MAX_MASK_WIDTH];
  for (int i = 0; i < MAX_MASK_WIDTH; i++) h_mask[i] = 1; 

  srand(123);
  for (int i = 0; i < input_width; i++) {
    a[i] = rand() % 256;
  }

  T *d_a, *d_b;
  #pragma omp target enter data map(to:a[0:size_bytes], b[0:size_bytes])

  #pragma omp target data map(to:h_mask[0:mask_width])
  {
    #pragma omp target map(from:b[0:size_bytes])
    {
      #pragma omp target teams distribute parallel for
      for (int i = 0; i < repeat; i++) {
        conv1d <<< grids, blocks >>> (d_a, d_b, input_width, mask_width);
      }

      #pragma omp target teams distribute parallel for
      for (int i = 0; i < repeat; i++) {
        conv1d_tiled <<< grids, blocks >>> (d_a, d_b, input_width, mask_width);
      }

      #pragma omp target teams distribute parallel for
      for (int i = 0; i < repeat; i++) {
        conv1d_tiled_caching <<< grids, blocks >>> (d_a, d_b, input_width, mask_width);
      }
    }
  }

  #pragma omp target exit data map(release:a[0:size_bytes], b[0:size_bytes])

  reference(a, b, h_mask, input_width, mask_width);

  free(a);
  free(b);
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