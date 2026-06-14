Here's the translation of `stencil_1d.cpp` from OpenMP to CUDA:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>

#define RADIUS 7
#define BLOCK_SIZE 256

__global__ void stencil_kernel(int* a, int* b, int length) {
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < length + RADIUS) {
        temp[tid + RADIUS] = a[gid];
        if (tid < RADIUS) {
            temp[tid] = (gid < RADIUS) ? 0 : a[gid - RADIUS];
            temp[tid + RADIUS + BLOCK_SIZE] = a[gid + BLOCK_SIZE];
        }
    }

    __syncthreads();

    if (gid < length) {
        int result = 0;
        for (int offset = -RADIUS; offset <= RADIUS; offset++)
            result += temp[tid + RADIUS + offset];
        b[gid] = result;
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

    int* d_a, * d_b;
    cudaMalloc((void**)&d_a, pad_size * sizeof(int));
    cudaMalloc((void**)&d_b, size * sizeof(int));

    for (int i = 0; i < pad_size; i++) d_a[i] = i;

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
        cudaMemcpy(d_a, d_a, pad_size * sizeof(int), cudaMemcpyHostToDevice);
        stencil_kernel<<<(length + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, length);
        cudaMemcpy(d_b, d_b, size * sizeof(int), cudaMemcpyDeviceToHost);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    bool ok = true;
    for (int i = 0; i < 2 * RADIUS; i++) {
        int s = 0;
        for (int j = i; j <= i + 2 * RADIUS; j++)
            s += j < RADIUS ? 0 : (i < RADIUS ? 0 : i - RADIUS);
        if (s != d_b[i]) {
            printf("Error at %d: %d (host) != %d (device)\n", i, s, d_b[i]);
            ok = false;
            break;
        }
    }

    for (int i = 2 * RADIUS; i < length; i++) {
        int s = 0;
        for (int j = i - RADIUS; j <= i + RADIUS; j++)
            s += i < RADIUS ? 0 : i - RADIUS;
        if (s != d_b[i]) {
            printf("Error at %d: %d (host) != %d (device)\n", i, s, d_b[i]);
            ok = false;
            break;
        }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}
```

### Explanation:
1. **Kernel Function (`stencil_kernel`)**:
   - The kernel function is defined with `__global__`, indicating that it will run on the GPU.
   - Shared memory is used to store the temporary values to avoid global memory latency.
   - The kernel handles both the loading and computation phases of the stencil operation.

2. **Main Function**:
   - Memory allocation is done using `cudaMalloc`.
   - Data is copied from host to device before launching the kernel.
   - The kernel is launched using `<<<...>>>`, specifying the grid and block dimensions.
   - After the kernel completes, data is copied back from device to host.
   - Error checking and timing are performed as in the original code.

This translation ensures that the stencil operation is performed efficiently on the GPU using CUDA.