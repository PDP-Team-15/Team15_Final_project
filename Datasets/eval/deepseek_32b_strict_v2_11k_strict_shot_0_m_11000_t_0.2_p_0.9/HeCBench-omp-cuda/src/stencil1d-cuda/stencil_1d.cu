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

void computeStencil(int* a, int* b, int length, int repeat) {
    int pad_size = length + RADIUS;
    int* d_a;
    int* d_b;
    int* d_temp;

    cudaMalloc((void**)&d_a, pad_size * sizeof(int));
    cudaMalloc((void**)&d_b, length * sizeof(int));
    cudaMalloc((void**)&d_temp, (BLOCK_SIZE + 2 * RADIUS) * sizeof(int));

    cudaMemcpy(d_a, a, pad_size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, 1, 1);
    dim3 blocks(1, 1, 1);

    for (int i = 0; i < repeat; i++) {
        for (int block_start = 0; block_start < length; block_start += BLOCK_SIZE) {
            int* temp = d_temp;

            cudaMemcpy(temp, d_a + block_start, BLOCK_SIZE * sizeof(int), cudaMemcpyDeviceToDevice);

            if (block_start < RADIUS) {
                for (int j = 0; j < RADIUS; j++) {
                    int idx = j;
                    if (block_start - RADIUS + idx < 0) {
                        temp[idx] = 0;
                    } else {
                        temp[idx] = d_a[block_start - RADIUS + idx];
                    }
                }
                for (int j = BLOCK_SIZE; j < BLOCK_SIZE + RADIUS; j++) {
                    int idx = j - BLOCK_SIZE;
                    if (block_start + j < pad_size) {
                        temp[j] = d_a[block_start + j];
                    } else {
                        temp[j] = 0;
                    }
                }
            }

            kernelStencil<<<blocks, threads>>>(d_a, temp, block_start, length, pad_size);
            cudaMemcpy(b + block_start, d_b + block_start, BLOCK_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_temp);
}

__global__ void kernelStencil(int* a, int* temp, int block_start, int length, int pad_size) {
    int j = threadIdx.x;
    int gindex = block_start + j;

    if (gindex >= length) return;

    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++) {
        result += temp[j + RADIUS + offset];
    }
    a[gindex] = result;
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
    int pad_size = length + RADIUS;

    int* a = (int*)malloc(pad_size * sizeof(int));
    int* b = (int*)malloc(size * sizeof(int));

    for (int i = 0; i < length + RADIUS; i++) a[i] = i;

    auto start = std::chrono::steady_clock::now();

    computeStencil(a, b, length, repeat);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    bool ok = true;
    for (int i = 0; i < 2 * RADIUS; i++) {
        int s = 0;
        for (int j = i; j <= i + 2 * RADIUS; j++)
            s += j < RADIUS ? 0 : (a[j] - RADIUS);
        if (s != b[i]) {
            printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
            ok = false;
            break;
        }
    }

    for (int i = 2 * RADIUS; i < length; i++) {
        int s = 0;
        for (int j = i - RADIUS; j <= i + RADIUS; j++)
            s += a[j];
        if (s != b[i]) {
            printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
            ok = false;
            break;
        }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    free(a);
    free(b);
    return 0;
}