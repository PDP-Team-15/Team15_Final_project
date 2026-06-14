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


__global__ void solution(double *u, const int n, const int nsteps, const double dt, const double alpha, const double length) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx < n && idy < n) {
    double x = (idx+1)*length/n;
    double y = (idy+1)*length/n;
    u[idx+idy*n] = exp(-2.0*alpha*PI*PI*dt/(length*length)) * sin(PI*x/length) * sin(PI*y/length);
  }
}

__global__ void l2norm(double *u, double *answer, const int n, const int nsteps, const double dt, const double alpha, const double length) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx < n && idy < n) {
    double x = (idx+1)*length/n;
    double y = (idy+1)*length/n;
    answer[idx+idy*n] = exp(-2.0*alpha*PI*PI*dt/(length*length)) * sin(PI*x/length) * sin(PI*y/length);
  }
}

__global__ void update(double *u, double *u_tmp, const int n, const double r2, const double r, const double dx, const double length) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx < n && idy < n) {
    u_tmp[idx+idy*n] = r2 * u[idx+idy*n] +
    r * ((idx < n-1) ? u[(idx+1)+idy*n] : 0.0) +
    r * ((idx > 0)   ? u[(idx-1)+idy*n] : 0.0) +
    r * ((idy < n-1) ? u[idx+(idy+1)*n] : 0.0) +
    r * ((idy > 0)   ? u[idx+(idy-1)*n] : 0.0);
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

  double tic, toc;
  const int block_size = 256;

  double *d_u, *d_u_tmp;
  cudaMalloc((void**)&d_u, sizeof(double)*n*n);
  cudaMalloc((void**)&d_u_tmp, sizeof(double)*n*n);

  cudaMemcpy(d_u, u, sizeof(double)*n*n, cudaMemcpyHostToDevice);

  dim3 dimGrid((n+block_size-1)/block_size, (n+block_size-1)/block_size, 1);
  dim3 dimBlock(block_size, block_size, 1);

  solution<<<dimGrid, dimBlock>>>(d_u, n, nsteps, dt, alpha, length);
  cudaDeviceSynchronize();

  for (int t = 0; t < nsteps; ++t) {
    cudaMemcpy(d_u_tmp, u, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    update<<<dimGrid, dimBlock>>>(d_u, d_u_tmp, n, 1.0 - 4.0*r, r, dx, length);
    cudaDeviceSynchronize();
    cudaMemcpy(u, d_u_tmp, sizeof(double)*n*n, cudaMemcpyDeviceToHost);
    double *tmp = u;
    u = u_tmp;
    u_tmp = tmp;
  }

  cudaMemcpy(d_u, u, sizeof(double)*n*n, cudaMemcpyHostToDevice);
  l2norm<<<dimGrid, dimBlock>>>(d_u, u, n, nsteps, dt, alpha, length);
  double *d_answer;
  cudaMalloc((void**)&d_answer, sizeof(double)*n*n);
  cudaMemcpy(d_answer, u, sizeof(double)*n*n, cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();

  double norm = 0.0;
  cudaMemcpy(u, d_answer, sizeof(double)*n*n, cudaMemcpyDeviceToHost);
  for (int i = 0; i < n*n; ++i) {
    norm += (u[i] - u[i]) * (u[i] - u[i]);
  }
  norm = sqrt(norm);
  cudaFree(d_answer);

  toc = omp_get_wtime();

  printf("Results\n\n");
  printf("Error (L2norm): %E\n", norm);
  printf("Solve time (s): %lf\n", toc-tic);
  printf("Total time (s): %lf\n", toc-start);
  printf("Bandwidth (GB/s): %lf\n", 1.0E-9*2.0*n*n*nsteps*sizeof(double)/(toc-tic));
  printf(LINE);

  free(u);
  free(u_tmp);
  cudaFree(d_u);
  cudaFree(d_u_tmp);

  return 0;
}

double solution(const double t, const double x, const double y, const double alpha, const double length) {

  return exp(-2.0*alpha*PI*PI*t/(length*length)) * sin(PI*x/length) * sin(PI*y/length);

}

double l2norm(const int n, const double * u, const int nsteps, const double dt,
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