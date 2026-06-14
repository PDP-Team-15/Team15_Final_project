#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <chrono>
#include <omp.h>
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

__global__ void benchmark_kernel(float *cd, int grid_dim, int block_dim, int compute_iterations, int idx, int stride, int big_stride) {
  const unsigned int blockSize = block_dim;
  const int k = blockIdx.x;
  const int j = threadIdx.x;
  const int i = blockIdx.y;

  float tmps[granularity];
  for(int k_loop=0; k_loop<fusion_degree; k_loop++) {
    tmps[j] = cd[idx+j*stride+k_loop*big_stride];

    for(int i_loop=0; i_loop<compute_iterations; i_loop++)
      tmps[j] = tmps[j]*tmps[j]+(float)seed;

    if (j < granularity) {
      __shared__ float shared[granularity];
      shared[j] = tmps[j];
      __syncthreads();

      if (j == 0) {
        float sum = 0;
        for(int j_loop=0; j_loop<granularity; j_loop+=2)
          sum += shared[j_loop]*shared[j_loop+1];

        for(int j_loop=0; j_loop<granularity; j_loop++)
          cd[idx+k_loop*big_stride] = sum;
      }
    }
  }
}

void mixbenchGPU(long size, int repeat) {
  const char *benchtype = "compute with global memory (block strided)";
  printf("Trade-off type:%s\n", benchtype);
  float *cd;
  cudaMalloc((void **)&cd, size*sizeof(float));
  cudaMemset(cd, 0, size*sizeof(float));

  const long reduced_grid_size = size/granularity/128;
  const int block_dim = 256;
  const int grid_dim = reduced_grid_size/block_dim;

  dim3 block(block_dim, 1, 1);
  dim3 grid(grid_dim, fusion_degree, 1);

  for (int i = 0; i < repeat; i++) {
    int idx = i;
    int stride = block_dim;
    int big_stride = grid_dim*block_dim*granularity;
    benchmark_kernel<<<grid, block>>>(cd, grid_dim, block_dim, i, idx, stride, big_stride);
    cudaDeviceSynchronize();
  }

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    int idx = i;
    int stride = block_dim;
    int big_stride = grid_dim*block_dim*granularity;
    benchmark_kernel<<<grid, block>>>(cd, grid_dim, block_dim, i, idx, stride, big_stride);
    cudaDeviceSynchronize();
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time: %f (s)\n", time * 1e-9f);

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