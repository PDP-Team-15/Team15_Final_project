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
#include <omp.h>

float distance_host(int i, float latitude_1, float longitude_1,
                   float latitude_2, float longitude_2) {
  // ... (same as in the original CUDA code)
}

void kernel_distance(const float4 *__restrict__ d_A,
                     float *__restrict__ d_C,
                     const int N) {
  #pragma omp parallel for
  for (int wiID = 0; wiID < N; wiID++) {
    // ... (same as in the original CUDA code, but without threadIdx and blockIdx)
  }
}

void distance_device(const float4* VA, float* VC, const size_t N, const int iteration) {
  float4 *d_VA = (float4*)malloc(sizeof(float4) * N);
  memcpy(d_VA, VA, sizeof(float4) * N);
  float *d_VC = (float*)malloc(sizeof(float) * N);

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < iteration; n++) {
    kernel_distance(d_VA, d_VC, N);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / iteration);

  memcpy(VC, d_VC, sizeof(float) * N);
  free(d_VA);
  free(d_VC);
}

void verify(int size, const float *output, const float *expected_output) {
  // ... (same as in the original CUDA code)
}

int main(int argc, char** argv) {
  // ... (same as in the original CUDA code)
}