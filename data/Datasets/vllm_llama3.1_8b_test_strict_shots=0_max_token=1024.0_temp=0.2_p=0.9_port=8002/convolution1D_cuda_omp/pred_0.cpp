```cpp
#include <omp.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <iostream>

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
  int i;
  #pragma omp parallel for
  for (i = 0; i < input_width; i++) {
    T s = 0;
    int n = mask_width / 2;
    int halo_left = (omp_get_thread_num() - n);
    int halo_right = (omp_get_thread_num() + n);
    T tile[TILE_SIZE + MAX_MASK_WIDTH - 1];
    for (int j = 0; j < TILE_SIZE + MAX_MASK_WIDTH - 1; j++) {
      if (halo_left <= halo_right && (halo_left <= j && j <= halo_right)) {
        if (halo_left < 0) {
          tile[j] = 0;
        } else if (halo_left >= input_width) {
          tile[j] = 0;
        } else {
          tile[j] = in[halo_left];
        }
      } else {
        tile[j] = in[i];
      }
    }
    for (int j = 0; j < mask_width; j++) {
      s += tile[omp_get_thread_num() + j] * mask[j];
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
  int i;
  #pragma omp parallel for
  for (i = 0; i < input_width; i++) {
    T s = 0;
    int this_tile_start = i / BLOCK_SIZE * BLOCK_SIZE;
    int next_tile_start = (i / BLOCK_SIZE + 1) * BLOCK_SIZE;
    int start = i - (mask_width / 2);
    T tile[TILE_SIZE];
    for (int j = 0; j < TILE_SIZE; j++) {
      tile[j] = in[i];
    }
    for (int j = 0; j < mask_width; j++) {
      int in_index = start + j;
      if (in_index >= 0 && in_index < input_width) {
        if (in_index >= this_tile_start && in_index < next_tile_start) {
          s += tile[j - (mask_width / 2)] * mask[j];
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
  std::cout << (ok ? "PASS" : "FAIL") << std::endl;
}

template <typename T>
void conv1D(const int input_width, const int mask_width, const int repeat)
{
  size_t size_bytes = input_width * sizeof(T);

  T *a, *b;
  a = (T *)malloc(size_bytes); 

  b = (T *)malloc(size_bytes); 


  T h_mask[MAX_MASK_WIDTH];

  for (int i = 