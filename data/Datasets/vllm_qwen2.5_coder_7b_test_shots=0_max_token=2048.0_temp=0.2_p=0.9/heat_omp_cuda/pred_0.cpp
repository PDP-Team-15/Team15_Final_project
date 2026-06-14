Here is the CUDA translation of the `heat.cpp` code:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define PI acos(-1.0) 
#define LINE "--------------------\n"

__device__ double solution(const double t, const double x, const double y, const double alpha, const double length) {
    return exp(-2.0 * alpha * PI * PI * t / (length * length)) * sin(PI * x / length) * sin(PI * y / length);
}

__global__ void initialize_u(double *u, int n, double dx, double length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < n && idy < n) {
        double x = (idx + 1) * dx;
        double y = (idy + 1) * dx;
        u[idx + idy * n] = sin(PI * x / length) * sin(PI * y / length);
    }
}

__global__ void initialize_u_tmp(double *u_tmp, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < n && idy < n) {
        u_tmp[idx + idy * n] = 0.0;
    }
}

__global__ void update_u(double *u, double *u_tmp, int n, double r2, double r) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < n && idy < n) {
        u_tmp[idx + idy * n] = r2 * u[idx + idy * n];
        if (idx < n - 1) u_tmp[idx + idy * n] += r * u[(idx + 1) + idy * n];
        if (idx > 0) u_tmp[idx + idy * n] += r * u[(idx - 1) + idy * n];
        if (idy < n - 1) u_tmp[idx + idy * n] += r * u[idx + (idy + 1) * n];
        if (idy > 0) u_tmp[idx + idy * n] += r * u[idx + (idy - 1) * n];
    }
}

double l2norm(const int n, const double *u, const int nsteps, const double dt, const double alpha, const double dx, const double length) {
    double time = dt * (double)nsteps;
    double l2norm = 0.0;
    double y = dx;
    for (int j = 0; j < n; ++j) {
        double x = dx;
        for (int i = 0; i < n; ++i) {
            double answer = solution(time, x, y, alpha, length);
            l2norm += (u[i + j * n] - answer) * (u[i + j * n] - answer);
            x += dx;
        }
        y += dx;
    }
    return sqrt(l2norm);
}

int main(int argc, char *argv[]) {
    double start = cudaGetRealtime();

    int n = 1000;
    int nsteps = 10;

    if (argc == 3) {
        n = atoi(argv[1]);
        if (n < 0) {
            fprintf(stderr, "Error: n must be positive\n");
            exit(EXIT_FAILURE);
        }

        nsteps = atoi(argv[2]);
        if (nsteps < 0) {
            fprintf(stderr, "Error: nsteps must be positive\n");
            exit(EXIT_FAILURE);
        }
    }

    double alpha = 0.1;
    double length = 1000.0;
    double dx = length / (n + 1);
    double dt = 0.5 / nsteps;

    double r = alpha * dt / (dx * dx);

    printf("\n");
    printf(" MMS heat equation\n\n");
    printf(LINE);
    printf("Problem input\n\n");
    printf(" Grid size: %d x %d\n", n, n);
    printf(" Cell width: %E\n", dx);
    printf(" Grid length: %lf x %lf\n", length, length);
    printf("\n");
    printf(" Alpha: %E\n", alpha);
    printf("\n");
    printf(" Steps: %d\n", nsteps);
    printf(" Total time: %E\n", dt * (double)nsteps);
    printf(" Time step: %E\n", dt);
    printf(LINE);

    printf("Stability\n\n");
    printf(" r value: %lf\n", r);
    if (r > 0.5)
        printf(" Warning: unstable\n");
    printf(LINE);

    double *u, *u_tmp;
    cudaMalloc((void **)&u, sizeof(double) * n * n);
    cudaMalloc((void **)&u_tmp, sizeof(double) * n * n);

    dim3 threadsPerBlock(256, 1);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initialize_u<<<blocksPerGrid, threadsPerBlock>>>(u, n, dx, length);
    initialize_u_tmp<<<blocksPerGrid, threadsPerBlock>>>(u_tmp, n);

    const double r2 = 1.0 - 4.0 * r;

    double tic, toc;
    tic = cudaGetRealtime();

    for (int t = 0; t < nsteps; ++t) {
        update_u<<<blocksPerGrid, threadsPerBlock>>>(u, u_tmp, n, r2, r);
        double *tmp = u;
        u = u_tmp;
        u_tmp = tmp;
    }

    toc = cudaGetRealtime();

    double norm = l2norm(n, u, nsteps, dt, alpha, dx, length);

    double stop = cudaGetRealtime();

    printf("Results\n\n");
    printf("Error (L2norm): %E\n", norm);
    printf("Solve time (s): %lf\n", toc - tic);
    printf("Total time (s): %lf\n", stop - start);
    printf("Bandwidth (GB/s): %lf\n", 1.0E-9 * 2.0 * n * n * nsteps * sizeof(double) / (toc - tic));
    printf(LINE);

    cudaFree(u);
    cudaFree(u_tmp);

    return 0;
}
```

This CUDA version includes the necessary device functions and kernels to perform the same operations as the original OpenMP code. The `initialize_u`, `initialize_u_tmp`, and `update_u` kernels handle the initialization and updating of the grid values, respectively. The `l2norm` function remains unchanged as it is a host-side function.