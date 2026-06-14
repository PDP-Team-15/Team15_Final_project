#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define PI acos(-1.0) 
#define LINE "--------------------\n"

double solution(const double t, const double x, const double y, const double alpha, const double length);
double l2norm(const int n, const double * __restrict u, const int nsteps, const double dt, const double alpha, const double dx, const double length);

__global__ void initKernel(double *u, int n, double dx, double length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < n && idy < n) {
        double x = (idx + 1) * dx;
        double y = (idy + 1) * dx;
        u[idx + idy * n] = sin(PI * x / length) * sin(PI * y / length);
    }
}

__global__ void solveKernel(double *u, double *u_tmp, int n, double r) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < n && idy < n) {
        double val = r2 * u[idx + idy * n];
        if (idx < n - 1) val += r * u[idx + 1 + idy * n];
        if (idx > 0) val += r * u[idx - 1 + idy * n];
        if (idy < n - 1) val += r * u[idx + (idy + 1) * n];
        if (idy > 0) val += r * u[idx + (idy - 1) * n];
        u_tmp[idx + idy * n] = val;
    }
}

int main(int argc, char *argv[]) {
    cudaEvent_t start, stop, tic, toc;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&tic);
    cudaEventCreate(&toc);
    
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
    
    const int block_size = 256;
    dim3 block(block_size, block_size);
    dim3 grid((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);
    
    double *u, *u_tmp;
    cudaMalloc((void**)&u, sizeof(double) * n * n);
    cudaMalloc((void**)&u_tmp, sizeof(double) * n * n);
    
    initKernel<<<grid, block>>>(u, n, dx, length);
    cudaDeviceSynchronize();
    
    double r2 = 1.0 - 4.0 * r;
    
    cudaEventRecord(tic);
    for (int t = 0; t < nsteps; ++t) {
        solveKernel<<<grid, block>>>(u, u_tmp, n, r);
        cudaDeviceSynchronize();
        
        double *temp = u;
        u = u_tmp;
        u_tmp = temp;
    }
    cudaEventRecord(toc);
    
    cudaEventSynchronize(toc);
    float solve_time;
    cudaEventElapsedTime(&solve_time, tic, toc);
    
    double *host_u;
    host_u = (double*)malloc(sizeof(double) * n * n);
    cudaMemcpy(host_u, u, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
    
    double norm = l2norm(n, host_u, nsteps, dt, alpha, dx, length);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float total_time;
    cudaEventElapsedTime(&total_time, start, stop);
    
    printf("Results\n\n");
    printf("Error (L2norm): %E\n", norm);
    printf("Solve time (s): %f\n", solve_time / 1000.0);
    printf("Total time (s): %f\n", total_time / 1000.0);
    printf("Bandwidth (GB/s): %lf\n", 1.0E-9 * 2.0 * n * n * nsteps * sizeof(double) / (solve_time / 1000.0));
    printf(LINE);
    
    free(host_u);
    cudaFree(u);
    cudaFree(u_tmp);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(tic);
    cudaEventDestroy(toc);
    
    return 0;
}

double solution(const double t, const double x, const double y, const double alpha, const double length) {
    return exp(-2.0 * alpha * PI * PI * t / (length * length)) * sin(PI * x / length) * sin(PI * y / length);
}

double l2norm(const int n, const double *u, const int nsteps, const double dt, const double alpha, const double dx, const double length) {
    double time = dt * nsteps;
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