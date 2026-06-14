```cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define PI acos(-1.0) 

#define LINE "--------------------\n" 


double solution(const double t, const double x, const double y, const double alpha, const double length);
double l2norm(const int n, const double * __restrict u, const int nsteps, const double dt, const double alpha, const double dx, const double length);

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

  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
  cudaEventRecord(start_event, 0);

  dim3 dimGrid((n+block_size-1)/block_size, (n+block_size-1)/block_size);
  dim3 dimBlock(block_size, block_size);

  cudaMemcpy(u, u, n*n*sizeof(double), cudaMemcpyHostToDevice);

  solution_kernel<<<dimGrid, dimBlock>>>(n, u, dx, length);

  cudaMemcpy(u, u, n*n*sizeof(double), cudaMemcpyDeviceToHost);

  solution_kernel<<<dimGrid, dimBlock>>>(n, u, dx, length);

  cudaMemcpy(u, u, n*n*sizeof(double), cudaMemcpyDeviceToHost);

  for (int t = 0; t < nsteps; ++t) {

    solution_kernel<<<dimGrid, dimBlock>>>(n, u, dx, length);

    double *tmp = u;
    u = u_tmp;
    u_tmp = tmp;
  }

  cudaMemcpy(u, u, n*n*sizeof(double), cudaMemcpyDeviceToHost);

  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);
  cudaEventElapsedTime(&toc, start_event, stop_event);

  double norm = l2norm(n, u, nsteps, dt, alpha, dx, length);

  double stop = omp_get_wtime();

  printf("Results\n\n");
  printf("Error (L2norm): %E\n", norm);
  printf("Solve time (s): %lf\n", toc/1000.0);
  printf("Total time (s): %lf\n", stop-start);
  printf("Bandwidth (GB/s): %lf\n", 1.0E-9*2.0*n*n*nsteps*sizeof(double)/(toc/1000.0));
  printf(LINE);

  free(u);
  free(u_tmp);
  cudaFreeDeviceMemory(u);
  cudaFreeDeviceMemory(u_tmp);

  return 0;
}

__global__ void solution_kernel(int n, double *u, double dx, double length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n) {
    double y = (j+1)*dx; 

    double x = (i+1)*dx; 

    u[i+j*n] = sin(PI * x / length) * sin(PI * y / length);
  }
}

__global__ void solution_kernel2(int n, double *u, double dx, double length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n) {
    u[i+j*n] = 0.0;
  }
}

__global__ void heat_kernel(int n, double *u, double *u_tmp, double r2, double r, double dx, double length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n) {
    u_tmp[i+j*n] =  r2 * u[i+j*n] +
    r * ((i < n-1) ? u[i+1+j*n] : 0.0) +
    r * ((i > 0)   ? u[i-1+j*n] : 0.0) +
    r * ((j < n-1) ? u[i+(j+1)*n] : 0.0) +
    r * ((j > 0)   ? u[i+(j-1)*n] : 0.0);
  }
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
```