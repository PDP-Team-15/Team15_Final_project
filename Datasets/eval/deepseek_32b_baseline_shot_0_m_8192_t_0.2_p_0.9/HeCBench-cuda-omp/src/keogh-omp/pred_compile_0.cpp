#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "reference.h"

typedef struct {
    float *cache;
    int start_idx;
    int end_idx;
} ThreadData;

void lb_keoghOMP(float *subject, float *avgs, float *stds, float *lb_keogh,
                 float *lower_bound, float *upper_bound, int M, int N) {
    const int num_threads = omp_get_max_threads();
    ThreadData *thread_data = (ThreadData *)malloc(num_threads * sizeof(ThreadData));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        thread_data[tid].cache = (float *)malloc((M + num_threads) * sizeof(float));
        thread_data[tid].start_idx = (N - M + 1) * tid / num_threads;
        thread_data[tid].end_idx = (N - M + 1) * (tid + 1) / num_threads;

        #pragma omp for
        for (int idx = 0; idx < N - M + 1; ++idx) {
            if (idx < thread_data[tid].start_idx || idx >= thread_data[tid].end_idx) {
                continue;
            }

            for (int k = 0; k < M; ++k) {
                if (idx + k < N) {
                    thread_data[tid].cache[k] = subject[idx + k];
                }
            }

            float residues = 0;
            float avg = avgs[idx];
            float std = stds[idx];

            for (int i = 0; i < M; ++i) {
                float value = (thread_data[tid].cache[i] - avg) / std;
                float lower = value - lower_bound[i];
                float upper = value - upper_bound[i];

                residues += upper * upper * (upper > 0) + lower * lower * (lower < 0);
            }

            lb_keogh[idx] = residues;
        }
    }

    for (int i = 0; i < num_threads; ++i) {
        free(thread_data[i].cache);
    }
    free(thread_data);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: ./%s <query length> <subject length> <repeat>\n", argv[0]);
        return 1;
    }

    const int M = atoi(argv[1]);
    const int N = atoi(argv[2]);
    const int repeat = atoi(argv[3]);

    printf("Query length = %d\n", M);
    printf("Subject length = %d\n", N);

    float *subject = (float*)malloc(sizeof(float)*N);
    float *lower = (float*)malloc(sizeof(float)*N);
    float *upper = (float*)malloc(sizeof(float)*N);
    float *lb = (float*)malloc(sizeof(float)*(N-M+1));
    float *lb_h = (float*)malloc(sizeof(float)*(N-M+1));
    float *avgs = (float*)malloc(sizeof(float)*(N-M+1));
    float *stds = (float*)malloc(sizeof(float)*(N-M+1));

    srand(123);
    for (int i = 0; i < N; ++i) subject[i] = (float)rand() / (float)RAND_MAX;
    for (int i = 0; i < N-M+1; ++i) avgs[i] = (float)rand() / (float)RAND_MAX;
    for (int i = 0; i < N-M+1; ++i) stds[i] = (float)rand() / (float)RAND_MAX;
    for (int i = 0; i < M; ++i) upper[i] = (float)rand() / (float)RAND_MAX;
    for (int i = 0; i < M; ++i) lower[i] = (float)rand() / (float)RAND_MAX;

    double start = omp_get_wtime();
    for (int i = 0; i < repeat; ++i) {
        lb_keoghOMP(subject, avgs, stds, lb, lower, upper, M, N);
    }
    double end = omp_get_wtime();
    double time = end - start;
    printf("Average kernel execution time: %f (s)\n", time / repeat);

    reference(subject, avgs, stds, lb_h, lower, upper, M, N);
    bool ok = true;
    for (int i = 0; i < N-M+1; ++i) {
        if (fabsf(lb[i] - lb_h[i]) > 1e-3f) {
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
    free(lower);
    free(upper);
    return 0;
}