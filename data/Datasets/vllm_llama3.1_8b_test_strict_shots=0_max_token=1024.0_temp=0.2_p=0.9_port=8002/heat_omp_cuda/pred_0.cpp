```cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define PI acos(-1.0) 

#define LINE "--------------------\n" 


__global__ void solution(double *u, const int n, const double t, const double alpha, const double length) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx < n && idy < n) {
    double x = (idx+1)*length/n;
    double y = (idy+1)*length/n;
    u[idx+idy*n] = exp(-2.0*alpha*PI*PI*t/(length*length)) * sin(PI*x/length) * sin(PI*y/length);
  }
}

__global__ void l2norm(double *u, double *answer, const int n, const int nsteps, const double dt,
                        const double alpha, const double dx, const double length) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx < n && idy < n) {
    double x = (idx+1)*dx;
    double y = (idy+1)*dx;
    answer[idx+idy*n] = exp(-2.0*alpha*PI*PI*dt*nsteps/(length*length)) * sin(PI*x/length) * sin(PI*y/length);
    double diff = u[idx+idy*n] - answer[idx+idy*n];
    atomicAdd(&diff, diff);
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
  double dx = length / (n+1);  
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
  printf(" Total time: %E\n", dt*(double)nsteps);
  printf(" Time step: %E\n", dt);
  printf(LINE);

  printf("Stability\n\n");
  printf(" r value: %lf\n", r);
  if (r > 0.5)
    printf(" Warning: unstable\n");
  printf(LINE);

  double *u     = (double*) malloc(sizeof(double)*n*n);
  double *u_tmp = (double*) malloc(sizeof(double)*n*n);
  double *answer = (double*) malloc(sizeof(double)*n*n);

  double tic, toc;
  const int block_size = 256;

  cudaMalloc((void**)&d_u, sizeof(double)*n*n);
  cudaMalloc((void**)&d_u_tmp, sizeof(double)*n*n);
  cudaMalloc((void**)&d_answer, sizeof(double)*n*n);

  cudaMemcpy(d_u, u, sizeof(double)*n*n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_u_tmp, u_tmp, sizeof(double)*n*n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_answer, answer, sizeof(double)*n*n, cudaMemcpyHostToDevice);

  dim3 dimGrid((n+block_size-1)/block_size, (n+block_size-1)/block_size);
  dim3 dimBlock(block_size, block_size);

  solution<<<dimGrid, dimBlock>>>(d_u, n, 0, alpha, length);
  l2norm<<<dimGrid, dimBlock>>>(d_u, d_answer, n, nsteps, dt, alpha, dx, length);

  cudaMemcpy(u, d_u, sizeof(double)*n*n, cudaMemcpyDeviceToHost);
  cudaMemcpy(u_tmp, d_u_tmp, sizeof(double)*n*n, cudaMemcpyDeviceToHost);
  cudaMemcpy(answer, d_answer, sizeof(double)*n*n, cudaMemcpyDeviceToHost);

  double r