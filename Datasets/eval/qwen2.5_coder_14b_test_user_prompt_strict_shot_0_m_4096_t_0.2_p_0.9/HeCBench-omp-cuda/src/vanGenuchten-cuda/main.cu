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

#define n 2.0
#define theta_S 1.0
#define theta_R 0.0
#define alpha 0.01

__global__ void vanGenuchten_kernel(
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
    m = lambda/n;

    _psi = psi[i] * 100.0;
    if (_psi < 0.0)
      _theta = (theta_S - theta_R) / pow(1.0 + pow((alpha*(-_psi)),n), m) + theta_R;
    else
      _theta = theta_S;

    theta[i] = _theta;

    Se = (_theta - theta_R)/(theta_S - theta_R);

    t = 1.0 - pow(1.0-pow(Se,1.0/m), m);
    K[i] = Ksat[i] * sqrt(Se) * t * t;

    if (_psi < 0.0)
      C[i] = 100 * alpha * n * (1.0/n-1.0)*pow(alpha*abs(_psi), n-1.0)
        * (theta_R-theta_S) * pow(pow(alpha*abs(_psi), n)+1.0, 1.0/n-2.0);
    else
      C[i] = 0.0;
  }
}

void vanGenuchten(
  const double *__restrict Ksat,
  const double *__restrict psi,
        double *__restrict C,
        double *__restrict theta,
        double *__restrict K,
  const int size)
{
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  vanGenuchten_kernel<<<blocksPerGrid, threadsPerBlock>>>(Ksat, psi, C, theta, K, size);
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
  
  cudaMallocManaged(&Ksat, size * sizeof(double));
  cudaMallocManaged(&psi, size * sizeof(double));
  cudaMallocManaged(&C, size * sizeof(double));
  cudaMallocManaged(&theta, size * sizeof(double));
  cudaMallocManaged(&K, size * sizeof(double));

  cudaMallocManaged(&C_ref, size * sizeof(double));
  cudaMallocManaged(&theta_ref, size * sizeof(double));
  cudaMallocManaged(&K_ref, size * sizeof(double));

  for (int i = 0; i < size; i++) {
    Ksat[i] = 1e-6 +  (1.0 - 1e-6) * i / size; 
    psi[i] = -100.0 + 101.0 * i / size;
  }

  reference(Ksat, psi, C_ref, theta_ref, K_ref, size);

  cudaDeviceSynchronize();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    vanGenuchten(Ksat, psi, C, theta, K, size);

  cudaDeviceSynchronize();

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

  cudaFree(Ksat);
  cudaFree(psi);
  cudaFree(C);
  cudaFree(theta);
  cudaFree(K);
  cudaFree(C_ref);
  cudaFree(theta_ref);
  cudaFree(K_ref);

  return 0;
}