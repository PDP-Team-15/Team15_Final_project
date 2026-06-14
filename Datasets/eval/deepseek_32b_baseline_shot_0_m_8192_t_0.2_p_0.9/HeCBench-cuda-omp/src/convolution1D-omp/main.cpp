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
#include <chrono>
#include <omp.h>

#define MAX_MASK_WIDTH 10
#define BLOCK_SIZE 256
#define TILE_SIZE BLOCK_SIZE

template<typename T>
T mask[MAX_MASK_WIDTH];

template<typename T>
void conv1d(const T *in, T *out, const int input_width, const int mask_width) {
    #pragma omp parallel for
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
void conv1d_tiled(const T *in, T *out, const int input_width, const int mask_width) {
    int n = mask_width / 2;
    int num_tiles = (input_width + BLOCK_SIZE - 1) / BLOCK_SIZE;

    #pragma omp parallel
    {
        T tile[TILE_SIZE + MAX_MASK_WIDTH - 1];
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        int block_start = (thread_id * BLOCK_SIZE);
        int block_end = block_start + BLOCK_SIZE;

        // Halo left
        if (thread_id > 0) {
            int halo_left = (thread_id - 1) * BLOCK_SIZE + (thread_id - 1) % BLOCK_SIZE;
            if (halo_left < 0) {
                tile[thread_id - (BLOCK_SIZE - n)] = 0;
            } else {
                tile[thread_id - (BLOCK_SIZE - n)] = in[halo_left];
            }
        }

        // Main tile
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (block_start + i < input_width) {
                tile[n + i] = in[block_start + i];
            } else {
                tile[n + i] = 0;
            }
        }

        // Halo right
        if (thread_id < num_threads - 1) {
            int halo_right = (thread_id + 1) * BLOCK_SIZE + (thread_id + 1) % BLOCK_SIZE;
            if (halo_right >= input_width) {
                tile[thread_id + BLOCK_SIZE + n] = 0;
            } else {
                tile[thread_id + BLOCK_SIZE + n] = in[halo_right];
            }
        }

        #pragma omp barrier

        for (int i = 0; i < BLOCK_SIZE; i++) {
            int global_i = block_start + i;
            if (global_i >= input_width) break;

            T s = 0;
            for (int j = 0; j < mask_width; j++) {
                s += tile[i + j] * mask<T>[j];
            }
            out[global_i] = s;
        }
    }
}

template<typename T>
void conv1d_tiled_caching(const T *in, T *out, const int input_width, const int mask_width) {
    int n = mask_width / 2;
    int num_tiles = (input_width + BLOCK_SIZE - 1) / BLOCK_SIZE;

    #pragma omp parallel
    {
        T tile[TILE_SIZE];
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        int block_start = thread_id * BLOCK_SIZE;
        int block_end = block_start + BLOCK_SIZE;

        // Load tile
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (block_start + i < input_width) {
                tile[i] = in[block_start + i];
            } else {
                tile[i] = 0;
            }
        }

        #pragma omp barrier

        for (int i = 0; i < BLOCK_SIZE; i++) {
            int global_i = block_start + i;
            if (global_i >= input_width) break;

            T s = 0;
            int start = global_i - n;
            for (int j = 0; j < mask_width; j++) {
                int in_index = start + j;
                if (in_index >= 0 && in_index < input_width) {
                    if (in_index >= block_start && in_index < block_end) {
                        s += tile[in_index - block_start] * mask<T>[j];
                    } else {
                        s += in[in_index] * mask<T>[j];
                    }
                }
            }
            out[global_i] = s;
        }
    }
}

template <typename T>
void reference(const T *h_in, const T *d_out, const T *mask, const int input_width, const int mask_width) {
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
void conv1D(const int input_width, const int mask_width, const int repeat) {
    size_t size_bytes = input_width * sizeof(T);

    T *a = (T *)malloc(size_bytes);
    T *b = (T *)malloc(size_bytes);

    T h_mask[MAX_MASK_WIDTH];
    for (int i = 0; i < MAX_MASK_WIDTH; i++) h_mask[i] = 1;

    srand(123);
    for (int i = 0; i < input_width; i++) {
        a[i] = rand() % 256;
    }

    // Initialize mask
    for (int i = 0; i < mask_width; i++) {
        mask<T>[i] = h_mask[i];
    }

    // Warm-up
    conv1d(a, b, input_width, mask_width);

    // Benchmark conv1d
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
        conv1d(a, b, input_width, mask_width);
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time of conv1d kernel: %f (us)\n",
           (time * 1e-3f) / repeat);
    reference(a, b, h_mask, input_width, mask_width);

    // Benchmark conv1d_tiled
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
        conv1d_tiled(a, b, input_width, mask_width);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time of conv1d-tiled kernel: %f (us)\n",
           (time * 1e-3f) / repeat);
    reference(a, b, h_mask, input_width, mask_width);

    // Benchmark conv1d_tiled_caching
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
        conv1d_tiled_caching(a, b, input_width, mask_width);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time of conv1d-tiled-caching kernel: %f (us)\n",
           (time * 1e-3f) / repeat);
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