#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>
#include "reference.h"

__global__ void vanGenuchten(
  const double *__restrict Ksat,
  const double *__restrict psi,
        double *__restrict C,
        double *__restrict theta,
        double *__restrict K,
  const int size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    double Se, _theta, _psi, lambda, m, t;

    lambda = n - 1.0;
    m = lambda / n;

    _psi = psi[i] * 100.0;
    if (_psi < 0.0)
      _theta = (theta_S - theta_R) / pow(1.0 + pow((alpha * (-_psi)), n), m) + theta_R;
    else
      _theta = theta_S;

    theta[i] = _theta;

    Se = (_theta - theta_R) / (theta_S - theta_R);

    t = 1.0 - pow(1.0 - pow(Se, 1.0 / m), m);
    K[i] = Ksat[i] * sqrt(Se) * t * t;

    if (_psi < 0.0)
      C[i] = 100 * alpha * n * (1.0 / n - 1.0) * pow(alpha * abs(_psi), n - 1.0)
        * (theta_R - theta_S) * pow(pow(alpha * abs(_psi), n) + 1.0, 1.0 / n - 2.0);
    else
      C[i] = 0.0;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: ./%s <dimX> <dimY> <dimZ> <repeat>\n", argv[0]);
    return 1;
  }

  const int dimX = atoi(argv[1]);
  const int dimY = atoi(argv[2]);
  const int dimZ = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const int size = dimX * dimY * dimZ;

  double *Ksat, *psi, *C, *theta, *K;
  double *C_ref, *theta_ref, *K_ref;

  // Allocate host memory
  Ksat = (double*)malloc(size * sizeof(double));
  psi = (double*)malloc(size * sizeof(double));
  C = (double*)malloc(size * sizeof(double));
  theta = (double*)malloc(size * sizeof(double));
  K = (double*)malloc(size * sizeof(double));

  C_ref = (double*)malloc(size * sizeof(double));
  theta_ref = (double*)malloc(size * sizeof(double));
  K_ref = (double*)malloc(size * sizeof(double));

  // Initialize input data
  for (int i = 0; i < size; i++) {
    Ksat[i] = 1e-6 + (1.0 - 1e-6) * i / size;
    psi[i] = -100.0 + 101.0 * i / size;
  }

  // Allocate device memory
  double *d_Ksat, *d_psi, *d_C, *d_theta, *d_K;
  cudaMalloc((void**)&d_Ksat, size * sizeof(double));
  cudaMalloc((void**)&d_psi, size * sizeof(double));
  cudaMalloc((void**)&d_C, size * sizeof(double));
  cudaMalloc((void**)&d_theta, size * sizeof(double));
  cudaMalloc((void**)&d_K, size * sizeof(double));

  // Copy input data to device
  cudaMemcpy(d_Ksat, Ksat, size * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_psi, psi, size * sizeof(double), cudaMemcpyHostToDevice);

  // Reference computation
  reference(Ksat, psi, C_ref, theta_ref, K_ref, size);

  // Kernel configuration
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    vanGenuchten<<<blocksPerGrid, threadsPerBlock>>>(d_Ksat, d_psi, d_C, d_theta, d_K, size);
    cudaDeviceSynchronize();
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  // Copy results back to host
  cudaMemcpy(C, d_C, size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(theta, d_theta, size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(K, d_K, size * sizeof(double), cudaMemcpyDeviceToHost);

  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (fabs(C[i] - C_ref[i]) > 1e-3 ||
        fabs(theta[i] - theta_ref[i]) > 1e-3 ||
        fabs(K[i] - K_ref[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  // Cleanup
  free(Ksat);
  free(psi);
  free(C);
  free(theta);
  free(K);
  free(C_ref);
  free(theta_ref);
  free(K_ref);

  cudaFree(d_Ksat);
  cudaFree(d_psi);
  cudaFree(d_C);
  cudaFree(d_theta);
  cudaFree(d_K);

  return 0;
}