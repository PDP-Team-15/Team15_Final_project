#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <omp.h>

float distance_host(int i, float latitude_1, float longitude_1,
                    float latitude_2, float longitude_2) {
  // ... (same as original)
}

void compute_distance(const float4* A, float* C, const int N) {
  #pragma omp parallel for
  for (int wiID = 0; wiID < N; ++wiID) {
    // ... (same as kernel_distance without __global__ and CUDA specifics)
  }
}

void distance_device(const float4* VA, float* VC, const size_t N, const int iteration) {
  #pragma omp parallel
  {
    #pragma omp single
    {
      auto start = std::chrono::steady_clock::now();
      for (int n = 0; n < iteration; ++n) {
        compute_distance(VA, VC, N);
      }
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / iteration);
    }
  }
}

// ... (rest of the code remains the same)