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

__global__ void init_u_kernel(double *u, const int n, const double dx, const double length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < n && idy < n) {
        double x = (idx + 1) * dx;
        double y = (idy + 1) * dx;
        u[idx + idy * n] = sin(PI * x / length) * sin(PI * y / length);
    }
}

__global__ void init_u_tmp_kernel(double *u_tmp, const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < n && idy < n) {
        u_tmp[idx + idy * n] = 0.0;
    }
}

__global__ void time_step_kernel(double *u, double *u_tmp, const int n, const double r) {
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

__global__ void norm_kernel(const double *u, double *norm, const int n, const double dx, const double time, const double alpha, const double length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < n && idy < n) {
        double x = (idx + 1) * dx;
        double y = (idy + 1) * dx;
        double exact = solution(time, x, y, alpha, length);
        norm[idx + idy * n] = (u[idx + idy * n] - exact) * (u[idx + idy * n] - exact);
    }
}

int main(int argc, char *argv[]) {
    cudaEvent_t start, stop, tic, toc;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&tic);
    cudaEventCreate(&toc);
    
    cudaEventRecord(start, 0);
    
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
    
    double *u, *u_tmp;
    cudaMalloc(&u, sizeof(double) * n * n);
    cudaMalloc(&u_tmp, sizeof(double) * n * n);
    
    dim3 block(256, 1, 1);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y, 1);
    
    init_u_kernel<<<grid, block>>>(u, n, dx, length);
    init_u_tmp_kernel<<<grid, block>>>(u_tmp, n);
    
    cudaEventRecord(tic, 0);
    
    for (int t = 0; t < nsteps; ++t) {
        time_step_kernel<<<grid, block>>>(u, u_tmp, n, r);
        
        double *temp = u;
        u = u_tmp;
        u_tmp = temp;
    }
    
    cudaEventRecord(toc, 0);
    cudaEventSynchronize(toc);
    
    double norm = 0.0;
    double *d_norm;
    cudaMalloc(&d_norm, sizeof(double) * n * n);
    
    norm_kernel<<<grid, block>>>(u, d_norm, n, dx, dt * nsteps, alpha, length);
    
    cudaMemcpy(&norm, d_norm, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
    norm = sqrt(norm);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float solve_time, total_time;
    cudaEventElapsedTime(&solve_time, tic, toc);
    cudaEventElapsedTime(&total_time, start, stop);
    
    printf("Results\n\n");
    printf("Error (L2norm): %E\n", norm);
    printf("Solve time (s): %f\n", solve_time / 1000.0);
    printf("Total time (s): %f\n", total_time / 1000.0);
    printf("Bandwidth (GB/s): %f\n", (2.0 * n * n * nsteps * sizeof(double)) / (solve_time / 1000.0) / 1e9);
    printf(LINE);
    
    cudaFree(u);
    cudaFree(u_tmp);
    cudaFree(d_norm);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(tic);
    cudaEventDestroy(toc);
    
    return 0;
}

double solution(const double t, const double x, const double y, const double alpha, const double length) {
    return exp(-2.0 * alpha * PI * PI * t / (length * length)) * sin(PI * x / length) * sin(PI * y / length);
}