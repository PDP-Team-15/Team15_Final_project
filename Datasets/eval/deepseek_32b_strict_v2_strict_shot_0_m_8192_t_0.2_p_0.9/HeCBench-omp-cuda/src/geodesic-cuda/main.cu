#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

typedef struct __attribute__((__aligned__(16)))
{
  float x;
  float y;
  float z;
  float w;
} float4;

float distance_host(int i, float latitude_1, float longitude_1,
                    float latitude_2, float longitude_2)
{
  // ... (same as original code)
}

__global__ void compute_distance_kernel(const float4* VA, float* VC, const size_t N,
                                       const int iteration)
{
  const int threadsPerBlock = 256;
  const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  for (int n = 0; n < iteration; n++) {
    for (int wiID = blockIdx.x * threadsPerBlock + threadIdx.x; wiID < N; wiID += blocks * threadsPerBlock) {
      // ... (same as original distance computation code)
      VC[wiID] = dist;
    }
  }
}

void distance_device(const float4* h_VA, float* h_VC, const size_t N, const int iteration)
{
  float4* d_VA;
  float* d_VC;
  size_t size = N * sizeof(float4);
  size_t sizeFloat = N * sizeof(float);

  cudaMalloc((void**)&d_VA, size);
  cudaMalloc((void**)&d_VC, sizeFloat);

  cudaMemcpy(d_VA, h_VA, size, cudaMemcpyHostToDevice);

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < iteration; n++) {
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    compute_distance_kernel<<<blocks, threadsPerBlock>>>(d_VA, d_VC, N, iteration);
    cudaDeviceSynchronize();
  }

  cudaMemcpy(h_VC, d_VC, sizeFloat, cudaMemcpyDeviceToHost);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / iteration);

  cudaFree(d_VA);
  cudaFree(d_VC);
}

void verify(int size, const float* output, const float* expected_output)
{
  // ... (same as original code)
}

int main(int argc, char** argv)
{
  // ... (same as original code)
  distance_device(input, output, N, iteration);
  verify(N, output, expected_output);
  // ... (same as original code)
}