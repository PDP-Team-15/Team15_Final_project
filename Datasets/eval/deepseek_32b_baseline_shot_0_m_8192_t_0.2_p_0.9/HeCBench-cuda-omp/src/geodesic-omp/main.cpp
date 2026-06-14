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

struct float4 {
    float x, y, z, w;
};

float distance_host(int i, float latitude_1, float longitude_1,
                   float latitude_2, float longitude_2)
{
    // ... (same as in the CUDA code)
}

void kernel_distance_omp(const float4* d_A, float* d_C, const int N)
{
    #pragma omp parallel
    {
        int wiID = omp_get_thread_num();
        if (wiID >= N) return;

        // ... (same as in the CUDA kernel)
    }
}

void distance_device(const float4* VA, float* VC, const size_t N, const int iteration)
{
    omp_set_num_threads(256);

    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < iteration; n++) {
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            kernel_distance_omp(VA, VC, N);
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / iteration);
}

// ... (rest of the code remains the same)