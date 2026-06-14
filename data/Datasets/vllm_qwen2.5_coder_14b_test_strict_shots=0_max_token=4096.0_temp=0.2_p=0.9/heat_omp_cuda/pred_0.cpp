```cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define PI acos(-1.0) 

#define LINE "--------------------\n" 

__device__ double solution(const double t, const double x, const double y, const double alpha, const double length);
__device__ double l2norm(const int n, const double * __restrict u, const int nsteps, const double dt, const double alpha, const double dx, const double length);

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

  double *u, *u_tmp;
  cudaMallocManaged(&u, sizeof(double)*n*n);
  cudaMallocManaged(&u_tmp, sizeof(double)*n*n);

  double tic, toc;
  const int block_size = 256;

  dim3 threadsPerBlock(block_size, block_size);
  dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

  cudaMemPrefetchAsync(u, sizeof(double)*n*n, cudaCpuDeviceId);
  cudaMemPrefetchAsync(u_tmp, sizeof(double)*n*n, cudaCpuDeviceId);

  initialize<<<numBlocks, threadsPerBlock>>>(n, u, dx, length);
  initialize<<<numBlocks, threadsPerBlock>>>(n, u_tmp, dx, length);

  const double r2 = 1.0 - 4.0*r;

  tic = omp_get_wtime();

  for (int t = 0; t < nsteps; ++t) {
    update<<<numBlocks, threadsPerBlock>>>(n, u, u_tmp, r, r2);
    double *tmp = u;
    u = u_tmp;
    u_tmp = tmp;
  }

  toc = omp_get_wtime();

  double norm = 0.0;
  l2norm<<<numBlocks, threadsPerBlock>>>(n, u, nsteps, dt, alpha, dx, length, &norm);
  cudaMemcpyFromSymbol(&norm, norm, sizeof(double));

  double stop = omp_get_wtime();

  printf("Results\n\n");
  printf("Error (L2norm): %E\n", norm);
  printf("Solve time (s): %lf\n", toc-tic);
  printf("Total time (s): %lf\n", stop-start);
  printf("Bandwidth (GB/s): %lf\n", 1.0E-9*2.0*n*n*nsteps*sizeof(double)/(toc-tic));
  printf(LINE);

  cudaFree(u);
  cudaFree(u_tmp);
}

__global__ void initialize(const int n, double *u, const double dx, const double length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < n) {
    double y = (j+1)*dx; 
    double x = (i+1)*dx; 
    u[i+j*n] = sin(PI * x / length) * sin(PI * y / length);
  }
}

__global__ void update(const int n, double *u, double *u_tmp, const double r, const double r2) {
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

__global__ void l2norm(const int n, const double *u, const int nsteps, const double dt,
                      const double alpha, const double dx, const double length, double *norm) {
  extern __shared__ double sdata[];
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  double sum = 0.0;
  if (i < n && j < n) {
    double time = dt * (double)nsteps;
    double x = dx * (i + 1);
    double y = dx * (j + 1);
    double answer = solution(time, x, y, alpha, length);
    sum = (u[i+j*n] - answer) * (u[i+j*n] - answer);
  }
  sdata[tid] = sum;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(norm, sdata[0]);
  }
}

__device__ double solution(const double t, const double x, const double y, const double alpha, const double length) {
  return exp(-2.0*alpha*PI*PI*t/(length*length)) * sin(PI*x/length) * sin(PI*y/length);
}

__device__ double l2norm(const int n, const double *u, const int nsteps, const double dt,
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