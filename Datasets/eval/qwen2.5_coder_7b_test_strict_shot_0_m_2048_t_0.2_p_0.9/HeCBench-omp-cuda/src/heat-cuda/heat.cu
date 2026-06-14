#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define PI acos(-1.0) 

#define LINE "--------------------\n" 

__global__ void initialize_u(double *u, int n, double dx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < n && idy < n) {
        double x = (idx + 1) * dx;
        double y = (idy + 1) * dx;
        u[idx + idy * n] = sin(PI * x) * sin(PI * y);
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
        int offset = idx + idy * n;
        u_tmp[offset] = r2 * u[offset];
        if (idx < n - 1) u_tmp[offset] += r * u[offset + 1];
        if (idx > 0) u_tmp[offset] += r * u[offset - 1];
        if (idy < n - 1) u_tmp[offset] += r * u[offset + n];
        if (idy > 0) u_tmp[offset] += r * u[offset - n];
    }
}

double solution(const double t, const double x, const double y, const double alpha, const double length) {
    return exp(-2.0 * alpha * PI * PI * t / (length * length)) * sin(PI * x / length) * sin(PI * y / length);
}

double l2norm(const int n, const double *u, const int nsteps, const double dt, const double alpha, const double dx, const double length) {
    double time = dt * static_cast<double>(nsteps);
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
        n = std::atoi(argv[1]);
        if (n < 0) {
            std::cerr << "Error: n must be positive\n";
            return EXIT_FAILURE;
        }

        nsteps = std::atoi(argv[2]);
        if (nsteps < 0) {
            std::cerr << "Error: nsteps must be positive\n";
            return EXIT_FAILURE;
        }
    }

    double alpha = 0.1;
    double length = 1000.0;
    double dx = length / (n + 1);
    double dt = 0.5 / nsteps;

    double r = alpha * dt / (dx * dx);

    std::cout << "\n";
    std::cout << " MMS heat equation\n\n";
    std::cout << LINE;
    std::cout << "Problem input\n\n";
    std::cout << " Grid size: " << n << " x " << n << "\n";
    std::cout << " Cell width: " << dx << "\n";
    std::cout << " Grid length: " << length << " x " << length << "\n";
    std::cout << "\n";
    std::cout << " Alpha: " << alpha << "\n";
    std::cout << "\n";
    std::cout << " Steps: " << nsteps << "\n";
    std::cout << " Total time: " << dt * static_cast<double>(nsteps) << "\n";
    std::cout << " Time step: " << dt << "\n";
    std::cout << LINE;

    std::cout << "Stability\n\n";
    std::cout << " r value: " << r << "\n";
    if (r > 0.5)
        std::cout << " Warning: unstable\n";
    std::cout << LINE;

    double *u, *u_tmp;
    cudaMalloc((void **)&u, sizeof(double) * n * n);
    cudaMalloc((void **)&u_tmp, sizeof(double) * n * n);

    dim3 threadsPerBlock(256, 1);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initialize_u<<<blocksPerGrid, threadsPerBlock>>>(u, n, dx);
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

    std::cout << "Results\n\n";
    std::cout << "Error (L2norm): " << norm << "\n";
    std::cout << "Solve time (s): " << toc - tic << "\n";
    std::cout << "Total time (s): " << stop - start << "\n";
    std::cout << "Bandwidth (GB/s): " << 1.0E-9 * 2.0 * n * n * nsteps * sizeof(double) / (toc - tic) << "\n";
    std::cout << LINE;

    cudaFree(u);
    cudaFree(u_tmp);

    return 0;
}