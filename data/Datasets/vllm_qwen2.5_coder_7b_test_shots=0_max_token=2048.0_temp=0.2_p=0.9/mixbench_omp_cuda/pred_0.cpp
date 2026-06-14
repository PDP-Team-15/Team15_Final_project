Here is the translation of the provided OpenMP code to CUDA:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

#define VECTOR_SIZE (8*1024*1024)
#define granularity (8)
#define fusion_degree (4)
#define seed 0.1f

__global__ void benchmark_func_kernel(float *cd, int grid_dim, int block_dim, int compute_iterations) {
  const unsigned int blockSize = blockDim.x;
  const int stride = blockSize;
  int idx = blockIdx.x * blockDim.x * granularity + threadIdx.x;
  const int big_stride = gridDim.x * blockDim.x * granularity;
  float tmps[granularity];
  
  for(int k = 0; k < fusion_degree; k++) {
    for(int j = 0; j < granularity; j++) {
      tmps[j] = cd[idx + j * stride + k * big_stride];
      
      for(int i = 0; i < compute_iterations; i++)
        tmps[j] = tmps[j] * tmps[j] + (float)seed;
    }

    float sum = 0;
    for(int j = 0; j < granularity; j += 2)
      sum += tmps[j] * tmps[j + 1];

    for(int j = 0; j < granularity; j++)
      cd[idx + k * big_stride] = sum;
  }
}

void mixbenchGPU(long size, int repeat) {
  const char *benchtype = "compute with global memory (block strided)";
  printf("Trade-off type:%s\n", benchtype);
  float *cd_h = (float*) malloc(size * sizeof(float));
  for (int i = 0; i < size; i++) cd_h[i] = 0;

  float *cd_d;
  cudaMalloc((void**)&cd_d, size * sizeof(float));
  cudaMemcpy(cd_d, cd_h, size * sizeof(float), cudaMemcpyHostToDevice);

  const long reduced_grid_size = size / granularity / 128;
  const int block_dim = 256;
  const int grid_dim = reduced_grid_size / block_dim;

  for (int i = 0; i < repeat; i++) {
    benchmark_func_kernel<<<grid_dim, block_dim>>>(cd_d, grid_dim, block_dim, i);
  }

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    benchmark_func_kernel<<<grid_dim, block_dim>>>(cd_d, grid_dim, block_dim, i);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time: %f (s)\n", time * 1e-9f);

  cudaMemcpy(cd_h, cd_d, size * sizeof(float), cudaMemcpyDeviceToHost);

  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (cd_h[i] != 0) {
      if (fabsf(cd_h[i] - 0.050807f) > 1e-6f) {
        ok = false;
        printf("Verification failed at index %d: %f\n", i, cd_h[i]);
        break;
      }
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(cd_h);
  cudaFree(cd_d);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  unsigned int datasize = VECTOR_SIZE * sizeof(float);

  printf("Buffer size: %dMB\n", datasize / (1024 * 1024));

  mixbenchGPU(VECTOR_SIZE, repeat);

  return 0;
}
```

### Explanation:
1. **Kernel Function**: The `benchmark_func` function has been converted into a CUDA kernel named `benchmark_func_kernel`. It uses CUDA's grid and block indexing to manage threads.
2. **Memory Management**: Host memory (`cd_h`) is allocated using `malloc`, and device memory (`cd_d`) is allocated using `cudaMalloc`. Data is copied between host and device using `cudaMemcpy`.
3. **Grid and Block Dimensions**: The grid and block dimensions are set based on the problem size and the desired performance characteristics.
4. **Execution Timing**: The timing logic remains similar to the original OpenMP version, measuring the total execution time of the kernel.
5. **Verification**: The verification step checks the results against expected values.

This CUDA code should provide equivalent functionality to the original OpenMP code while leveraging GPU parallelism.