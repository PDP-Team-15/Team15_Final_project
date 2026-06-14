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
#include <chrono>

#define VECTOR_SIZE (8*1024*1024)
#define granularity (8)
#define fusion_degree (4)
#define seed 0.1f

#pragma omp declare target
void benchmark_func(float *g_data,
                   const int compute_iterations)
{
  const unsigned int blockSize = blockDim.x;
  const int stride = blockSize;
  int idx = blockIdx.x*blockSize*granularity + threadIdx.x;
  const int big_stride = gridDim.x*blockSize*granularity;

  float tmps[granularity];
  for(int k=0; k<fusion_degree; k++) {
    #pragma unroll
    for(int j=0; j<granularity; j++) {
      

      tmps[j] = g_data[idx+j*stride+k*big_stride];

      

      for(int i=0; i<compute_iterations; i++)
        tmps[j] = tmps[j]*tmps[j]+seed;
    }

    

    float sum = 0.f;
    #pragma unroll
    for(int j=0; j<granularity; j+=2)
      sum += tmps[j]*tmps[j+1];

    #pragma unroll
    for(int j=0; j<granularity; j++)
      g_data[idx+k*big_stride] = sum;
  }
}

#pragma omp declare target
void mixbenchGPU(long size, int repeat) {
  const char *benchtype = "compute with global memory (block strided)";
  printf("Trade-off type:%s\n", benchtype);
  float *cd = (float*) malloc (size*sizeof(float));
  for (int i = 0; i < size; i++) cd[i] = 0;

  const long reduced_grid_size = size/granularity/128;
  const int block_dim = 256;
  const int grid_dim = reduced_grid_size/block_dim;

  // float *d_cd;
  // cudaMalloc((void**)&d_cd, size*sizeof(float));
  // cudaMemcpy(d_cd, cd,  size*sizeof(float), cudaMemcpyHostToDevice);

  

  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams distribute parallel for collapse(2)
    for (int k = 0; k < fusion_degree; k++) {
      for (int j = 0; j < granularity; j++) {
        int idx = blockIdx.x*blockSize*granularity + threadIdx.x;
        const int big_stride = gridDim.x*blockSize*granularity;

        float tmps[granularity];
        tmps[j] = cd[idx+j*stride+k*big_stride];

        for(int i=0; i<compute_iterations; i++)
          tmps[j] = tmps[j]*tmps[j]+seed;

        float sum = 0.f;
        for(int j=0; j<granularity; j+=2)
          sum += tmps[j]*tmps[j+1];

        for(int j=0; j<granularity; j++)
          cd[idx+k*big_stride] = sum;
      }
    }
  }

  // cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams distribute parallel for collapse(2)
    for (int k = 0; k < fusion_degree; k++) {
      for (int j = 0; j < granularity; j++) {
        int idx = blockIdx.x*blockSize*granularity + threadIdx.x;
        const int big_stride = gridDim.x*blockSize*granularity;

        float tmps[granularity];
        tmps[j] = cd[idx+j*stride+k*big_stride];

        for(int i=0; i<compute_iterations; i++)
          tmps[j] = tmps[j]*tmps[j]+seed;

        float sum = 0.f;
        for(int j=0; j<granularity; j+=2)
          sum += tmps[j]*tmps[j+1];

        for(int j=0; j<granularity; j++)
          cd[idx+k*big_stride] = sum;
      }
    }
  }

  // cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time: %f (s)\n", time * 1e-9f);

  // cudaMemcpy(cd, d_cd, size*sizeof(float), cudaMemcpyDeviceToHost);

  

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

  free(cd);
  // cudaFree(d_cd);
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