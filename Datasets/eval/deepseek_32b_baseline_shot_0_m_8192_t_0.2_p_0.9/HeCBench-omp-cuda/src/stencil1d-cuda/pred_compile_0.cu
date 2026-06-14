#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>

#define RADIUS 7
#define BLOCK_SIZE 256

__global__ void stencil_kernel(int* a, int* b, int length, int pad_size) {
    int block_start = blockIdx.x * BLOCK_SIZE;
    int j = threadIdx.x;

    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];

    int gindex = block_start + j;

    // Load data into shared memory
    temp[j + RADIUS] = a[gindex];
    if (j < RADIUS) {
        if (gindex - RADIUS >= 0) {
            temp[j] = a[gindex - RADIUS];
        } else {
            temp[j] = 0;
        }
        if (gindex + BLOCK_SIZE < pad_size) {
            temp[j + RADIUS + BLOCK_SIZE] = a[gindex + BLOCK_SIZE];
        } else {
            temp[j + RADIUS + BLOCK_SIZE] = 0;
        }
    }

    __syncthreads();

    // Compute stencil
    if (gindex < length) {
        int result = 0;
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            result += temp[j + RADIUS + offset];
        }
        b[gindex] = result;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <length> <repeat>\n", argv[0]);
        printf("length is a multiple of %d\n", BLOCK_SIZE);
        return 1;
    }
    const int length = atoi(argv[1]);
    const int repeat = atoi(argv[2]);

    int size = length;
    int pad_size = (length + RADIUS);

    int* a = (int *)malloc(pad_size * sizeof(int));
    int* b = (int *)malloc(size * sizeof(int));

    for (int i = 0; i < length + RADIUS; i++) {
        a[i] = i;
    }

    int* d_a;
    int* d_b;
    cudaMalloc(&d_a, pad_size * sizeof(int));
    cudaMalloc(&d_b, size * sizeof(int));

    cudaMemcpy(d_a, a, pad_size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid((length + BLOCK_SIZE - 1) / BLOCK_SIZE);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
        stencil_kernel<<<grid, block>>>(d_a, d_b, length, pad_size);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(b, d_b, size * sizeof(int), cudaMemcpyDeviceToHost);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    bool ok = true;
    for (int i = 0; i < 2 * RADIUS; i++) {
        int s = 0;
        for (int j = i; j <= i + 2 * RADIUS; j++) {
            s += (j < RADIUS) ? 0 : (a[j] - RADIUS);
        }
        if (s != b[i]) {
            printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
            ok = false;
            break;
        }
    }

    for (int i = 2 * RADIUS; i < length; i++) {
        int s = 0;
        for (int j = i - RADIUS; j <= i + RADIUS; j++) {
            s += a[j];
        }
        if (s != b[i]) {
            printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
            ok = false;
            break;
        }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    free(a);
    free(b);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}