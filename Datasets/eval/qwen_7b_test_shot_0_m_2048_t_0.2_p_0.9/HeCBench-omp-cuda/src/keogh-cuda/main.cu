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

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void compute_residues(float *subject, float *avgs, float *stds, float *lower_bound, float *upper_bound, float *lb, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N - M + 1) return;

    float residues = 0;
    float avg = avgs[idx];
    float std = stds[idx];

    for (int i = 0; i < M; ++i) {
        float value = (subject[idx + i] - avg) / std;
        float lower = value - lower_bound[i];
        float upper = value - upper_bound[i];

        residues += upper * upper * (upper > 0) + lower * lower * (lower < 0);
    }

    lb[idx] = residues;
}

void reference(float *subject, float *avgs, float *stds, float *lb_h, float *lower_bound, float *upper_bound, int M, int N);

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

    float *subject = (float*)malloc(sizeof(float) * N);
    float *lower_bound = (float*)malloc(sizeof(float) * N);
    float *upper_bound = (float*)malloc(sizeof(float) * N);
    float *lb = (float*)malloc(sizeof(float) * (N - M + 1));
    float *lb_h = (float*)malloc(sizeof(float) * (N - M + 1));
    float *avgs = (float*)malloc(sizeof(float) * (N - M + 1));
    float *stds = (float*)malloc(sizeof(float) * (N - M + 1));

    srand(123);
    for (int i = 0; i < N; ++i) subject[i] = (float)rand() / (float)RAND_MAX;
    for (int i = 0; i < N - M + 1; ++i) avgs[i] = (float)rand() / (float)RAND_MAX;
    for (int i = 0; i < N - M + 1; ++i) stds[i] = (float)rand() / (float)RAND_MAX;
    for (int i = 0; i < M; ++i) upper_bound[i] = (float)rand() / (float)RAND_MAX;
    for (int i = 0; i < M; ++i) lower_bound[i] = (float)rand() / (float)RAND_MAX;

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N - M + 1 + threadsPerBlock - 1) / threadsPerBlock;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    for (int i = 0; i < repeat; i++) {
        CHECK_CUDA(cudaMemcpyAsync(subject, subject, sizeof(float) * N, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(avgs, avgs, sizeof(float) * (N - M + 1), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(stds, stds, sizeof(float) * (N - M + 1), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(lower_bound, lower_bound, sizeof(float) * N, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(upper_bound, upper_bound, sizeof(float) * N, cudaMemcpyHostToDevice, stream));

        compute_residues<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(subject, avgs, stds, lower_bound, upper_bound, lb, M, N);

        CHECK_CUDA(cudaMemcpyAsync(lb_h, lb, sizeof(float) * (N - M + 1), cudaMemcpyDeviceToHost, stream));
    }

    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaStreamDestroy(stream));

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
        compute_residues<<<blocksPerGrid, threadsPerBlock>>>(subject, avgs, stds, lower_bound, upper_bound, lb, M, N);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    reference(subject, avgs, stds, lb_h, lower_bound, upper_bound, M, N);
    bool ok = true;
    for (int i = 0; i < N - M + 1; i++) {
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
    return 0;
}