#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>
#include "reference.h"

#define BLOCKS 256

__global__ void kernel(float* subject, float* avgs, float* stds, float* lower_bound, float* upper_bound, float* lb, int M, int N) {
    int idx = blockIdx.x;
    if (idx >= N - M + 1) return;

    float residues = 0.0f;
    float avg = avgs[idx];
    float std = stds[idx];

    __shared__ float s_residues[BLOCK_SIZE];
    s_residues[threadIdx.x] = 0.0f;

    for (int i = threadIdx.x; i < M; i += BLOCK_SIZE) {
        float value = (subject[idx + i] - avg) / std;
        float lower = value - lower_bound[i];
        float upper = value - upper_bound[i];

        residues += upper * upper * (upper > 0) + lower * lower * (lower < 0);
    }

    s_residues[threadIdx.x] = residues;
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum += s_residues[i];
        }
        lb[idx] = sum;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: ./%s <query length> <subject length> <repeat>\n", argv[0]);
        return -1;
    }

    const int M = atoi(argv[1]);
    const int N = atoi(argv[2]);
    const int repeat = atoi(argv[3]);

    printf("Query length = %d\n", M);
    printf("Subject length = %d\n", N);

    float *subject = (float*) malloc(sizeof(float) * N);
    float *lower_bound = (float*) malloc(sizeof(float) * N);
    float *upper_bound = (float*) malloc(sizeof(float) * N);
    float *lb = (float*) malloc(sizeof(float) * (N - M + 1));
    float *lb_h = (float*) malloc(sizeof(float) * (N - M + 1));
    float *avgs = (float*) malloc(sizeof(float) * (N - M + 1));
    float *stds = (float*) malloc(sizeof(float) * (N - M + 1));

    srand(123);
    for (int i = 0; i < N; ++i) subject[i] = (float)rand() / (float)RAND_MAX;
    for (int i = 0; i < N - M + 1; ++i) avgs[i] = (float)rand() / (float)RAND_MAX;
    for (int i = 0; i < N - M + 1; ++i) stds[i] = (float)rand() / (float)RAND_MAX;
    for (int i = 0; i < M; ++i) upper_bound[i] = (float)rand() / (float)RAND_MAX;
    for (int i = 0; i < M; ++i) lower_bound[i] = (float)rand() / (float)RAND_MAX;

    float *d_subject, *d_avgs, *d_stds, *d_lower_bound, *d_upper_bound, *d_lb;
    cudaMalloc(&d_subject, sizeof(float) * N);
    cudaMalloc(&d_avgs, sizeof(float) * (N - M + 1));
    cudaMalloc(&d_stds, sizeof(float) * (N - M + 1));
    cudaMalloc(&d_lower_bound, sizeof(float) * N);
    cudaMalloc(&d_upper_bound, sizeof(float) * N);
    cudaMalloc(&d_lb, sizeof(float) * (N - M + 1));

    cudaMemcpy(d_subject, subject, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_avgs, avgs, sizeof(float) * (N - M + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stds, stds, sizeof(float) * (N - M + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lower_bound, lower_bound, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_upper_bound, upper_bound, sizeof(float) * N, cudaMemcpyHostToDevice);

    const int grids = (N - M + 1 + BLOCKS - 1) / BLOCKS;

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
        kernel<<<grids, BLOCKS>>>(d_subject, d_avgs, d_stds, d_lower_bound, d_upper_bound, d_lb, M, N);
    }
    cudaMemcpy(lb, d_lb, sizeof(float) * (N - M + 1), cudaMemcpyDeviceToHost);
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    reference(subject, avgs, stds, lb_h, lower_bound, upper_bound, M, N);
    bool ok = true;
    for (int i = 0; i < N - M + 1; ++i) {
        if (fabs(lb[i] - lb_h[i]) > 1e-3) {
            printf("%d %f %f\n", i, lb[i], lb_h[i]);
            ok = false;
            break;
        }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    free(lb);
    free(lb_h);
    free(avgs);
    free(stds);
    free(subject);
    free(lower_bound);
    free(upper_bound);

    cudaFree(d_subject);
    cudaFree(d_avgs);
    cudaFree(d_stds);
    cudaFree(d_lower_bound);
    cudaFree(d_upper_bound);
    cudaFree(d_lb);

    return 0;
}