```cuda
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define PI acos(-1.0) 

#define LINE "--------------------\n" 


__device__ double solution(const double t, const double x, const double y, const double alpha, const double length);

__device__ double l2norm(const int n, const double * __restrict u, const int nsteps, const double dt, const double alpha, const double dx, const double length);

__global__ void initialize(double *u, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx < n && idy < n) {
    double y = (idy+1)*blockDim.y * gridDim.y * (double)dx; 
    double x = (idx+1)*blockDim.x * gridDim.x * (double)dx; 
    u[idx + idy * n] = sin(PI * x / length) * sin(PI * y / length);
  }
}

__global__ void initialize_tmp(double *u_tmp, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx < n && idy < n) {
    u_tmp[idx + idy * n] = 0.0;
  }
}

__global__ void update(double *u, double *u_tmp, int n, double r, double r2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx < n && idy < n) {
    u_tmp[idx + idy * n] =  r2 * u[idx + idy * n] +
    r * ((idx < n-1) ? u[(idx+1) + idy * n] : 0.0) +
    r * ((idx > 0)   ? u[(idx-1) + idy * n] : 0.0) +
    r * ((idy < n-1) ? u[idx + (idy+1) * n] : 0.0) +
    r * ((idy > 0)   ? u[idx + (idy-1) * n] : 0.0);
  }
}

int main(int argc, char *argv[]) {

  double start;

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

  double tic, toc;
  const int block_size = 256;

  cudaMalloc((void**)&u, n*n*sizeof(double));
  cudaMalloc((void**)&u_tmp, n*n*sizeof(double));

  cudaMemcpy(u, u, n*n*sizeof(double), cudaMemcpyHostToDevice);

  dim3 blockSize(block_size, block_size);
  dim3 gridSize((n+blockSize.x-1)/blockSize.x, (n+blockSize.y-1)/blockSize.y);

  initialize<<<gridSize, blockSize>>>(u, n);
  cudaDeviceSynchronize();

  initialize_tmp<<<gridSize, blockSize>>>(u_tmp, n);
  cudaDeviceSynchronize();

  const double r2 = 1.0 - 4.0*r;

  tic = omp_get_wtime();

  for (int t = 0; t < nsteps; ++t) {
    update<<<gridSize, blockSize>>>(u, u_tmp, n, r, r2);
    cudaDeviceSynchronize();
    double *tmp = u;
    u = u_tmp;
    u_tmp = tmp;
  }

  cudaMemcpy(u, u, n*n*sizeof(double), cudaMemcpyDeviceToHost);

  toc = omp_get_wtime();

  double norm = l2norm(n, u, nsteps, dt, alpha, dx, length);

  double stop = omp_get_wtime();

  printf("Results\n\n");
  printf("Error (L2norm): %E\n", norm);
  printf("Solve time (s): %lf\n", toc-tic);
  printf("Total time (s): %lf\n", stop-start);
  printf("Bandwidth (GB/s): %lf\n", 1.0E-9*2.0*n*n*nsteps*sizeof(double)/(toc-tic));
  printf(LINE);

  free(u);
  free(u_tmp);
  cudaFree(u);
  cudaFree(u_tmp);
}

__device__ double solution(const double t, const double x, const double y, const double alpha, const double length) {
  return exp(-2.0*alpha*PI*PI*t/(length*length)) * sin(PI*x/length) * sin(PI*y/length);
}

__device__ double l2norm(const int n, const double * __restrict u, const int nsteps, const double dt,
                         const double alpha, const double dx, const double length) {
  double time = dt * (double)nsteps;

  double l2norm = 0.0;

  double y = dx;
  for (int j = 0; j < n; ++j) {
    double x = dx;
    for (int i = 0; i < n; ++i) {
      double answer = solution(time, x, y, alpha, length);
      l2norm += (u[i+j*n] - answer) * (u[i+j*n] - answer);

      x += dx;
    }
    y += dx;
  }

  return sqrt(l2norm);
}
```