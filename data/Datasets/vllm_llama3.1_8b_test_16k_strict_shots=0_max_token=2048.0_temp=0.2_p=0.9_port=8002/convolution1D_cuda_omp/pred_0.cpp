```cpp
#include <omp.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define MAX_MASK_WIDTH 10
#define BLOCK_SIZE 256
#define TILE_SIZE BLOCK_SIZE

template<typename T>
T mask[MAX_MASK_WIDTH];

template<typename T>
void conv1d(const T * __restrict__ in,
                  T * __restrict__ out,
            const int input_width,
            const int mask_width)
{
  int i;
  #pragma omp parallel for
  for (i = 0; i < input_width; i++) {
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
void conv1d_tiled(const T *__restrict__ in,
                        T *__restrict__ out,
                  const int input_width,
                  const int mask_width)
{
  int i, j, k;
  int n = mask_width / 2;
  int halo_left, halo_right;
  T tile[TILE_SIZE + MAX_MASK_WIDTH - 1];
  #pragma omp parallel for private(i, j, k, halo_left, halo_right)
  for (i = 0; i < input_width; i++) {
    int block_idx = i / BLOCK_SIZE;
    int thread_idx = i % BLOCK_SIZE;
    int this_tile_start = block_idx * BLOCK_SIZE;
    int next_tile_start = (block_idx + 1) * BLOCK_SIZE;
    int start = i - (mask_width / 2);
    tile[thread_idx] = in[i];
    #pragma omp barrier
    if (thread_idx >= BLOCK_SIZE - n) {
      halo_left = (block_idx - 1) * BLOCK_SIZE + thread_idx;
      if (halo_left < 0) halo_left = 0;
      tile[thread_idx - (BLOCK_SIZE - n)] = in[halo_left];
    }
    if (thread_idx < n) {
      halo_right = (block_idx + 1) * BLOCK_SIZE + thread_idx;
      if (halo_right >= input_width) halo_right = input_width - 1;
      tile[thread_idx + BLOCK_SIZE + n] = in[halo_right];
    }
    #pragma omp barrier
    T s = 0;
    for (j = 0; j < mask_width; j++) {
      s += tile[thread_idx + j] * mask[j];
    }
    out[i] = s;
  }
}

template<typename T>
void conv1d_tiled_caching(const T *__restrict__ in,
                                T *__restrict__ out,
                          const int input_width,
                          const int mask_width)
{
  int i, j, k;
  int n = mask_width / 2;
  int halo_left, halo_right;
  T tile[TILE_SIZE];
  #pragma omp parallel for private(i, j, k, halo_left, halo_right)
  for (i = 0; i < input_width; i++) {
    int block_idx = i / BLOCK_SIZE;
    int thread_idx = i % BLOCK_SIZE;
    int this_tile_start = block_idx * BLOCK_SIZE;
    int next_tile_start = (block_idx + 1) * BLOCK_SIZE;
    int start = i - (mask_width / 2);
    tile[thread_idx] = in[i];
    #pragma omp barrier
    T s = 0;
    for (j = 0; j < mask_width; j++) {
      int in_index = start + j;
      if (in_index >= 0 && in_index < input_width) {
        if (in_index >= this_tile_start && in_index < next_tile_start) {
          s += tile[thread_idx + j - (mask_width / 2)] * mask[j];
        } else {
          s += in[in_index] * mask[j];
        }
      }
    }
    out[i] = s;
  }
}

template <typename T>
void reference(const T *h_in,
               const T *h_out,
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
    if (fabs(s - h_out[i]) > 1e-3) {
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

  T h_mask[MAX_MASK_WIDTH];
  for (int i = 0; i < MAX_MASK_WIDTH; i++) h_mask[i] = 1; 

  srand(123);
  for (int i = 0; i < input_width; i++) {
    a[i] = rand() % 256;
  }

  #pragma omp parallel for
  for (int i = 0; i < repeat; i++) {
    conv1d(a, b, input_width, mask_width);
  }

  cudaMemcpy(b, b, size_bytes, cudaMemcpyHostToDevice);
  reference(a, b, h_mask, input_width, mask_width);

  #pragma omp parallel for
  for (int i = 0; i < repeat; i++) {
    conv1d_tiled(a, b, input_width, mask_width);
  }

  cudaMemcpy(b, b, size_bytes, cudaMemcpyHostToDevice);
  reference(a, b, h_mask, input_width, mask_width);

  #pragma omp parallel for
  for (int i = 0; i < repeat; i++) {
    conv1d_tiled_caching(a, b, input_width, mask_width);
  }

  cudaMemcpy(b, b, size_bytes, cudaMemcpyHostToDevice);
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
```