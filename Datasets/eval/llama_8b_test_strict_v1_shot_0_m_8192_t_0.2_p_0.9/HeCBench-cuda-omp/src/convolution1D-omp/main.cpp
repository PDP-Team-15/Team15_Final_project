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
  int i = omp_get_thread_num() + omp_get_num_threads() * omp_get_max_threads() / BLOCK_SIZE;
  T s = 0;
  int start = i - mask_width / 2;
  for (int j = 0; j < mask_width; j++) {
    if (start + j >= 0 && start + j < input_width) {
      s += in[start + j] * mask[j];
    }
  }
  out[i] = s;
}

template<typename T>
void conv1d_tiled(const T *__restrict__ in,
                        T *__restrict__ out,
                  const int input_width,
                  const int mask_width)
{
  __shared__ T tile[TILE_SIZE + MAX_MASK_WIDTH - 1];
  int i = omp_get_thread_num() + omp_get_num_threads() * omp_get_max_threads() / BLOCK_SIZE;

  int n = mask_width / 2;  

  int halo_left = (omp_get_thread_num() - BLOCK_SIZE + n) < 0 ? 0 : in[omp_get_thread_num() - BLOCK_SIZE + n];

  tile[n + omp_get_thread_num()] = in[omp_get_thread_num()];

  int halo_right = (omp_get_thread_num() + BLOCK_SIZE + n) >= input_width ? 0 : in[omp_get_thread_num() + BLOCK_SIZE + n];

  __syncthreads();

  T s = 0;
  for (int j = 0; j < mask_width; j++)
    s += tile[omp_get_thread_num() + j] * mask[j];

  out[i] = s;
}

template<typename T>
void conv1d_tiled_caching(const T *__restrict__ in,
                                T *__restrict__ out,
                          const int input_width,
                          const int mask_width)
{
  __shared__ T tile[TILE_SIZE];

  int i = omp_get_thread_num() + omp_get_num_threads() * omp_get_max_threads() / BLOCK_SIZE;
  tile[omp_get_thread_num()] = in[i];
  __syncthreads();

  int this_tile_start = omp_get_thread_num() * BLOCK_SIZE;
  int next_tile_start = (omp_get_thread_num() + 1) * BLOCK_SIZE;
  int start = i - (mask_width / 2);
  T s = 0;
  for (int j = 0; j < mask_width; j++) {
    int in_index = start + j;
    if (in_index >= 0 && in_index < input_width) {
      if (in_index >= this_tile_start && in_index < next_tile_start) {
        s += tile[omp_get_thread_num() + j - (mask_width / 2)] * mask[j];
      } else {
        s += in[in_index] * mask[j];
      }
    }
  }
  out[i] = s;
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


  T h_mask[MAX_MASK_WIDTH];

  for (int i = 0; i < MAX_MASK_WIDTH; i++) h_mask[i] = 1; 

  srand(123);
  for (int i = 0; i < input_width; i++) {
    a[i] = rand() % 256;
  }

  T *d_a, *d_b;
  d_a = (T *)malloc(size_bytes);
  d_b = (T *)malloc(size_bytes);

  for (int i = 0; i < input_width; i++) {
    d_a[i] = a[i];
  }

  for (int i = 0; i < MAX_MASK_WIDTH; i++) {
    d_b[i] = h_mask[i];
  }

  #pragma omp parallel for
  for (int i = 0; i < input_width; i++) {
    conv1d<T>(d_a, d_b, input_width, mask_width);
  }

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel for
    for (int i = 0; i < input_width; i++) {
      conv1d_tiled<T>(d_a, d_b, input_width, mask_width);
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv1d kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  for (int i = 0; i < input_width; i++) {
    b[i] = d_b[i];
  }
  reference(a, b, h_mask, input_width, mask_width);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel for
    for (int i = 0; i < input_width; i++) {
      conv1d_tiled_caching<T>(d_a, d_b, input_width, mask_width);
    }
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv1d-tiled kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  for (int i = 0; i < input_width; i++) {
    b[i] = d_b[i];
  }
  reference(a, b, h_mask, input_width, mask_width);

  free(a);
  free(b);
  free(d_a);
  free(d_b);
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