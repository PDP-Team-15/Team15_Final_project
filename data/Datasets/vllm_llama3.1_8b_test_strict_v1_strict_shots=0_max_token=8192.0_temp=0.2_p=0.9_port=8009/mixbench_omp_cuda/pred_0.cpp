```cuda
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>

#define VECTOR_SIZE (8*1024*1024)
#define granularity (8)
#define fusion_degree (4)
#define seed 0.1f

__global__ void benchmark_kernel(float *cd, int grid_dim, int block_dim, int compute_iterations) { 
  const unsigned int blockSize = blockDim.x;
  const int stride = blockSize;
  int idx = blockIdx.x*blockDim.x*granularity + threadIdx.x;
  const int big_stride = grid_dim*blockDim.x*granularity;
  float tmps[granularity];
  for(int k=0; k<fusion_degree; k++) {
    #pragma unroll
    for(int j=0; j<granularity; j++) {
      tmps[j] = cd[idx+j*stride+k*big_stride];
      for(int i=0; i<compute_iterations; i++)
        tmps[j] = tmps[j]*tmps[j]+(float)seed;
    }

    float sum = 0;
    #pragma unroll
    for(int j=0; j<granularity; j+=2)
      sum += tmps[j]*tmps[j+1];

    #pragma unroll
    for(int j=0; j<granularity; j++)
      cd[idx+k*big_stride] = sum;
  }
}

void mixbenchGPU(long size, int repeat) {
  const char *benchtype = "compute with global memory (block strided)";
  printf("Trade-off type:%s\n", benchtype);
  float *cd;
  cudaMalloc((void**)&cd, size*sizeof(float));
  for (int i = 0; i < size; i++) cd[i] = 0;

  const long reduced_grid_size = size/granularity/128;
  const int block_dim = 256;
  const int grid_dim = reduced_grid_size/block_dim;

  cudaMemset(cd, 0, size*sizeof(float));

  for (int i = 0; i < repeat; i++) {
    benchmark_kernel<<<grid_dim, block_dim>>>(cd, grid_dim, block_dim, i);
    cudaDeviceSynchronize();
  }

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    benchmark_kernel<<<grid_dim, block_dim>>>(cd, grid_dim, block_dim, i);
    cudaDeviceSynchronize();
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time: %f (s)\n", time * 1e-9f);

  cudaDeviceSynchronize();

  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (cd[i] != 0) {
      if (fabsf(cd[i] - 0.050807f) > 1e-6f) {
        ok = false;
        printf("Verification failed at index %d: %f\n", i, cd[i]);
        break;
      }
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

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

Note: This code assumes that the CUDA device has at least 256 threads per block and that the grid size is a multiple of 128. You may need to adjust these values depending on your specific hardware and use case.