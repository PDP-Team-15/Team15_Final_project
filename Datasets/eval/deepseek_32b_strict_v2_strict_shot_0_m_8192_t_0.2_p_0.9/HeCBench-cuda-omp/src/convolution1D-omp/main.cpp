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
    int num_threads = omp_get_max_threads();
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
    int num_threads = omp_get_max_threads();
    int n = mask_width / 2;
    #pragma omp parallel for
    for (int i = 0; i < input_width; i++) {
        int tile_start = (i / BLOCK_SIZE) * BLOCK_SIZE;
        int tile_end = tile_start + BLOCK_SIZE;
        T tile[TILE_SIZE + MAX_MASK_WIDTH - 1];
        
        int halo_left = i - n;
        if (halo_left >= 0) {
            tile[threadIdx.x - (blockDim.x - n)] = in[halo_left];
        }
        
        tile[n + threadIdx.x] = in[i];
        
        int halo_right = i + n;
        if (halo_right < input_width) {
            tile[threadIdx.x + blockDim.x + n] = in[halo_right];
        }
        
        T s = 0;
        for (int j = 0; j < mask_width; j++) {
            s += tile[threadIdx.x + j] * mask<T>[j];
        }
        out[i] = s;
    }
}

template<typename T>
void conv1d_tiled_caching(const T *in, T *out, const int input_width, const int mask_width) {
    int num_threads = omp_get_max_threads();
    int n = mask_width / 2;
    #pragma omp parallel for
    for (int i = 0; i < input_width; i++) {
        int tile_start = (i / BLOCK_SIZE) * BLOCK_SIZE;
        int tile_end = tile_start + BLOCK_SIZE;
        T tile[TILE_SIZE];
        
        int global_idx = i;
        tile[threadIdx.x] = in[global_idx];
        
        int start = i - n;
        T s = 0;
        for (int j = 0; j < mask_width; j++) {
            int in_index = start + j;
            if (in_index >= 0 && in_index < input_width) {
                if (in_index >= tile_start && in_index < tile_end) {
                    s += tile[threadIdx.x + j - n] * mask<T>[j];
                } else {
                    s += in[in_index] * mask<T>[j];
                }
            }
        }
        out[i] = s;
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
    
    for (int i = 0; i < repeat; i++) {
        conv1d(a, b, input_width, mask_width);
    }
    
    reference(a, b, h_mask, input_width, mask_width);
    
    for (int i = 0; i < repeat; i++) {
        conv1d_tiled(a, b, input_width, mask_width);
    }
    
    reference(a, b, h_mask, input_width, mask_width);
    
    for (int i = 0; i < repeat; i++) {
        conv1d_tiled_caching(a, b, input_width, mask_width);
    }
    
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