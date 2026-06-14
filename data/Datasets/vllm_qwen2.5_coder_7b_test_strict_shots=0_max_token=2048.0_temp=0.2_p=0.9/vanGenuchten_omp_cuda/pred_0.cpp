```cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void vanGenuchtenKernel(
  const double *Ksat,
  const double *psi,
        double *C,
        double *theta,
        double *K,
  const int size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

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

void vanGenuchten(
  const double *Ksat,
  const double *psi,
        double *C,
        double *theta,
        double *K,
  const int size)
{
  vanGenuchtenKernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(Ksat, psi, C, theta, K, size);
  cudaDeviceSynchronize();
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

  double *d_Ksat, *d_psi, *d_C, *d_theta, *d_K;
  double *h_C_ref, *h_theta_ref, *h_K_ref;
  
  cudaMalloc(&d_Ksat, size * sizeof(double));
  cudaMalloc(&d_psi, size * sizeof(double));
  cudaMalloc(&d_C, size * sizeof(double));
  cudaMalloc(&d_theta, size * sizeof(double));
  cudaMalloc(&d_K, size * sizeof(double));

  h_C_ref = new double[size];
  h_theta_ref = new double[size];
  h_K_ref = new double[size];

  for (int i = 0; i < size; i++) {
    d_Ksat[i] = 1e-6 + (1.0 - 1e-6) * i / size;
    d_psi[i] = -100.0 + 101.0 * i / size;
  }

  reference(d_Ksat, d_psi, h_C_ref, h_theta_ref, h_K_ref, size);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    vanGenuchten(d_Ksat, d_psi, d_C, d_theta, d_K, size);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  cudaMemcpy(h_C, d_C, size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_theta, d_theta, size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_K, d_K, size * sizeof(double), cudaMemcpyDeviceToHost);

  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (fabs(h_C[i] - h_C_ref[i]) > 1e-3 ||
        fabs(h_theta[i] - h_theta_ref[i]) > 1e-3 ||
        fabs(h_K[i] - h_K_ref[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  cudaFree(d_Ksat);
  cudaFree(d_psi);
  cudaFree(d_C);
  cudaFree(d_theta);
  cudaFree(d_K);
  delete[] h_C_ref;
  delete[] h_theta_ref;
  delete[] h_K_ref;

  return 0;
}
```