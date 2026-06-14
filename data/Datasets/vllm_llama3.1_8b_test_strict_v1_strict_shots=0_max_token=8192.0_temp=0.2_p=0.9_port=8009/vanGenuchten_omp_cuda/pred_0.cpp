```cuda
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "reference.h"

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      printf("CUDA error: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1); \
    } \
  } while (0)

#define THREADS_PER_BLOCK 256

__global__ void vanGenuchtenKernel(
  double *Ksat,
  double *psi,
  double *C,
  double *theta,
  double *K,
  int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) return;

  double Se, _theta, _psi, lambda, m, t;

  lambda = n - 1.0;
  m = lambda / n;

  _psi = psi[idx] * 100.0;
  if (_psi < 0.0)
    _theta = (theta_S - theta_R) / pow(1.0 + pow((alpha * (-_psi)), n), m) + theta_R;
  else
    _theta = theta_S;

  theta[idx] = _theta;

  Se = (_theta - theta_R) / (theta_S - theta_R);

  t = 1.0 - pow(1.0 - pow(Se, 1.0 / m), m);
  K[idx] = Ksat[idx] * sqrt(Se) * t * t;

  if (_psi < 0.0)
    C[idx] = 100 * alpha * n * (1.0 / n - 1.0) * pow(alpha * abs(_psi), n - 1.0)
      * (theta_R - theta_S) * pow(pow(alpha * abs(_psi), n) + 1.0, 1.0 / n - 2.0);
  else
    C[idx] = 0.0;
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

  CUDA_CHECK(cudaMalloc((void**)&Ksat, size * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&psi, size * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&C, size * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&theta, size * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&K, size * sizeof(double)));

  C_ref = new double[size];
  theta_ref = new double[size];
  K_ref = new double[size];

  for (int i = 0; i < size; i++) {
    Ksat[i] = 1e-6 + (1.0 - 1e-6) * i / size;
    psi[i] = -100.0 + 101.0 * i / size;
  }

  reference(Ksat, psi, C_ref, theta_ref, K_ref, size);

  double *d_Ksat, *d_psi, *d_C, *d_theta, *d_K;
  CUDA_CHECK(cudaMalloc((void**)&d_Ksat, size * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_psi, size * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_C, size * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_theta, size * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_K, size * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_Ksat, Ksat, size * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_psi, psi, size * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C, C, size * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_theta, theta, size * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_K, K, size * sizeof(double), cudaMemcpyHostToDevice));

  int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  vanGenuchtenKernel<<<blocks, THREADS_PER_BLOCK>>>(d_Ksat, d_psi, d_C, d_theta, d_K, size);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int i = 0; i < repeat; i++)
    vanGenuchtenKernel<<<blocks, THREADS_PER_BLOCK>>>(d_Ksat, d_psi, d_C, d_theta, d_K, size);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(C, d_C, size * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(theta, d_theta, size * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(K, d_K, size * sizeof(double), cudaMemcpyDeviceToHost));

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    vanGenuchtenKernel<<<blocks, THREADS_PER_BLOCK>>>(d_Ksat, d_psi, d_C, d_theta, d_K, size);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

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

  delete(C_ref);
  delete(theta_ref);
  delete(K_ref);

  CUDA_CHECK(cudaFree(d_Ksat));
  CUDA_CHECK(cudaFree(d_psi));
  CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaFree(d_theta));
  CUDA_CHECK(cudaFree(d_K));

  return 0;
}
```

Note: The code assumes that the `reference` function is defined in the `reference.h` header file and is implemented correctly. Also, the `n`, `alpha`, `theta_S`, and `theta_R` variables are assumed to be defined and initialized correctly.