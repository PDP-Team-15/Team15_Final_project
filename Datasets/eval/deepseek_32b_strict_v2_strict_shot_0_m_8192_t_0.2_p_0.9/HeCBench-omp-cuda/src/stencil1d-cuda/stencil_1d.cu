#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>
#include <chrono>

#define RADIUS 7
#define BLOCK_SIZE 256

__global__ void stencilKernel(int* a, int* b, int length, int RADIUS) {
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];

    int global_i = blockIdx.x * BLOCK_SIZE;
    int local_i = threadIdx.x;

    if (local_i < BLOCK_SIZE) {
        int gindex = global_i + local_i;
        temp[local_i + RADIUS] = a[gindex];
        if (local_i < RADIUS) {
            temp[local_i] = (gindex < RADIUS) ? 0 : a[gindex - RADIUS];
            temp[local_i + RADIUS + BLOCK_SIZE] = a[gindex + BLOCK_SIZE];
        }
    }

    __syncthreads();

    if (local_i < BLOCK_SIZE) {
        int sum = 0;
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            sum += temp[local_i + RADIUS + offset];
        }
        b[global_i + local_i] = sum;
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

    int pad_size = length + 2 * RADIUS;
    int* a = (int*)malloc(pad_size * sizeof(int));
    int* b = (int*)malloc(length * sizeof(int));

    for (int i = 0; i < pad_size; i++) {
        a[i] = i;
    }

    int* d_a;
    int* d_b;
    cudaMalloc(&d_a, pad_size * sizeof(int));
    cudaMalloc(&d_b, length * sizeof(int));

    cudaMemcpy(d_a, a, pad_size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((length + BLOCK_SIZE - 1) / BLOCK_SIZE);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
        stencilKernel<<<gridSize, blockSize>>>(d_a, d_b, length, RADIUS);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(b, d_b, length * sizeof(int), cudaMemcpyDeviceToHost);

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