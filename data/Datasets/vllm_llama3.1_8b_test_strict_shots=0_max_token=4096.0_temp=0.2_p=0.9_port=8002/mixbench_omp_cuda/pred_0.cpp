```cpp
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>

#define VECTOR_SIZE (8*1024*1024)
#define granularity (8)
#define fusion_degree (4)
#define seed 0.1f

__global__ void benchmark_func(float *cd, int grid_dim, int block_dim, int compute_iterations, int idx, int stride, int big_stride) {
  const unsigned int blockSize = block_dim;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int big_idx = blockIdx.y * grid_dim + blockIdx.x;
  const int big_stride_global = big_idx * blockSize * granularity;
  float tmps[granularity];
  for(int k=0; k<fusion_degree; k++) {
    for(int j=0; j<granularity; j++) {
      tmps[j] = cd[idx+j*stride+k*big_stride_global];
      for(int i=0; i<compute_iterations; i++)
        tmps[j] = tmps[j]*tmps[j]+(float)seed;
    }
    float sum = 0;
    for(int j=0; j<granularity; j+=2)
      sum += tmps[j]*tmps[j+1];
    for(int j=0; j<granularity; j++)
      cd[idx+k*big_stride_global] = sum;
  }
}

void mixbenchGPU(long size, int repeat) {
  const char *benchtype = "compute with global memory (block strided)";
  printf("Trade-off type:%s\n", benchtype);
  float *cd;
  cudaMalloc((void**)&cd, size*sizeof(float));
  cudaMemset(cd, 0, size*sizeof(float));

  const long reduced_grid_size = size/granularity/128;
  const int block_dim = 256;
  const int grid_dim = reduced_grid_size/block_dim;

  dim3 block(block_dim, 1, 1);
  dim3 grid((reduced_grid_size + block_dim - 1) / block_dim, grid_dim, 1);

  for (int i = 0; i < repeat; i++) {
    benchmark_func<<<grid, block>>>(cd, grid_dim, block_dim, i, 0, block_dim, block_dim*grid_dim);
  }

  cudaDeviceSynchronize();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    benchmark_func<<<grid, block>>>(cd, grid_dim, block_dim, i, 0, block_dim, block_dim*grid_dim);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time: %f (s)\n", time * 1e-9f);

  bool ok = true;
  float *cd_host;
  cudaMallocHost((void**)&cd_host, size*sizeof(float));
  cudaMemcpy(cd_host, cd, size*sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; i++) {
    if (cd_host[i] != 0) {
      if (fabsf(cd_host[i] - 0.050807f) > 1e-6f) {
        ok = false;
        printf("Verification failed at index %d: %f\n", i, cd_host[i]);
        break;
      }
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  cudaFreeHost(cd_host);
  cudaFree(cd);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  unsigned int datasize = VECTOR_SIZE*sizeof(float);

  printf("Buffer size: %dMB\n", datasize/(1024*1024));

  mixbenchGPU(VECTOR_SIZE, repeat);

  return 0;
}
```