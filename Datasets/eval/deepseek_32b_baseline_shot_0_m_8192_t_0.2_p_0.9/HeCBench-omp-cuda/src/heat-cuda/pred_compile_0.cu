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

__global__ void initializeU(double *u, int n, double dx, double length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        double x = (i + 1) * dx;
        double y = (j + 1) * dx;
        u[i + j * n] = sin(PI * x / length) * sin(PI * y / length);
    }
}

__global__ void initializeUTmp(double *u_tmp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        u_tmp[i + j * n] = 0.0;
    }
}

__global__ void computeTimeStep(double *u, double *u_tmp, int n, double r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        double val = r2 * u[i + j * n];
        if (i < n - 1) val += r * u[i + 1 + j * n];
        else val += 0.0;
        if (i > 0) val += r * u[i - 1 + j * n];
        else val += 0.0;
        if (j < n - 1) val += r * u[i + (j + 1) * n];
        else val += 0.0;
        if (j > 0) val += r * u[i + (j - 1) * n];
        else val += 0.0;
        u_tmp[i + j * n] = val;
    }
}

__global__ void computeL2Norm(double *u, double *norm, int n, double dx, double time, double alpha, double length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        double x = (i + 1) * dx;
        double y = (j + 1) * dx;
        double answer = solution(time, x, y, alpha, length);
        *norm += (u[i + j * n] - answer) * (u[i + j * n] - answer);
    }
}

int main(int argc, char *argv[]) {
    double start = omp_get_wtime();

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

    const int block_size = 256;
    const dim3 block(block_size, block_size, 1);
    const dim3 grid((n + block_size - 1) / block_size, (n + block_size - 1) / block_size, 1);

    double *u, *u_tmp;
    cudaMalloc(&u, sizeof(double) * n * n);
    cudaMalloc(&u_tmp, sizeof(double) * n * n);

    initializeU<<<grid, block>>>(u, n, dx, length);
    cudaDeviceSynchronize();

    initializeUTmp<<<grid, block>>>(u_tmp, n);
    cudaDeviceSynchronize();

    const double r2 = 1.0 - 4.0 * r;

    double tic = omp_get_wtime();

    for (int t = 0; t < nsteps; ++t) {
        computeTimeStep<<<grid, block>>>(u, u_tmp, n, r);
        cudaDeviceSynchronize();

        double *temp = u;
        u = u_tmp;
        u_tmp = temp;
    }

    double toc = omp_get_wtime();

    double *norm_dev;
    cudaMalloc(&norm_dev, sizeof(double));
    cudaMemcpy(norm_dev, &l2norm, sizeof(double), cudaMemcpyHostToDevice);

    computeL2Norm<<<grid, block>>>(u, norm_dev, n, dx, dt * nsteps, alpha, length);
    cudaDeviceSynchronize();

    double norm;
    cudaMemcpy(&norm, norm_dev, sizeof(double), cudaMemcpyDeviceToHost);
    norm = sqrt(norm);

    double stop = omp_get_wtime();

    printf("Results\n\n");
    printf("Error (L2norm): %E\n", norm);
    printf("Solve time (s): %lf\n", toc - tic);
    printf("Total time (s): %lf\n", stop - start);
    printf("Bandwidth (GB/s): %lf\n", 1.0E-9 * 2.0 * n * n * nsteps * sizeof(double) / (toc - tic));
    printf(LINE);

    cudaFree(u);
    cudaFree(u_tmp);
    cudaFree(norm_dev);

    return 0;
}

double solution(const double t, const double x, const double y, const double alpha, const double length) {
    return exp(-2.0 * alpha * PI * PI * t / (length * length)) * sin(PI * x / length) * sin(PI * y / length);
}

double l2norm(const int n, const double *u, const int nsteps, const double dt,
              const double alpha, const double dx, const double length) {
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