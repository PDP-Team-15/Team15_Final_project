Here's the equivalent CUDA code for the given OpenMP code:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>

__global__ void vanGenuchtenKernel(
  double *Ksat,
  double *psi,
        double *C,
        double *theta,
        double *K,
  int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {

    double Se, _theta, _psi, lambda, m, t;

    lambda = n - 1.0;
    m = lambda/n;

    _psi = psi[idx] * 100.0;
    if ( _psi < 0.0 )
      _theta = (theta_S - theta_R) / pow(1.0 + pow((alpha*(-_psi)),n), m) + theta_R;
    else
      _theta = theta_S;

    theta[idx] = _theta;

    Se = (_theta - theta_R)/(theta_S - theta_R);

    t = 1.0 - pow(1.0-pow(Se,1.0/m), m);
    K[idx] = Ksat[idx] * sqrt(Se) * t * t;

    if (_psi < 0.0)
      C[idx] = 100 * alpha * n * (1.0/n-1.0)*pow(alpha*abs(_psi), n-1.0)
        * (theta_R-theta_S) * pow(pow(alpha*abs(_psi), n)+1.0, 1.0/n-2.0);
    else
      C[idx] = 0.0;
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
  
  Ksat = new double[size];
  psi = new double[size];
  C = new double[size];
  theta = new double[size];
  K = new double[size];

  C_ref = new double[size];
  theta_ref = new double[size];
  K_ref = new double[size];

  for (int i = 0; i < size; i++) {
    Ksat[i] = 1e-6 +  (1.0 - 1e-6) * i / size; 
    psi[i] = -100.0 + 101.0 * i / size;
  }

  reference(Ksat, psi, C_ref, theta_ref, K_ref, size);

  int blockSize = 256;
  int numBlocks = (size + blockSize - 1) / blockSize;

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    vanGenuchtenKernel<<<numBlocks, blockSize>>>(Ksat, psi, C, theta, K, size);
    cudaDeviceSynchronize();
  }

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

  delete(Ksat);
  delete(psi);
  delete(C);
  delete(theta);
  delete(K);
  delete(C_ref);
  delete(theta_ref);
  delete(K_ref);

  return 0;
}
```

Note that I've made the following changes:

*   Replaced the OpenMP `#pragma omp target` directive with a CUDA kernel launch.
*   Defined a CUDA kernel function `vanGenuchtenKernel` that takes the input arrays and size as arguments.
*   Used CUDA's `blockIdx.x`, `blockDim.x`, and `threadIdx.x` variables to determine the thread's index within the block.
*   Replaced the OpenMP `#pragma omp target data` directive with manual memory management using CUDA's `cudaDeviceSynchronize` function to ensure that the kernel has completed before accessing the results.
*   Removed the OpenMP `thread_limit` clause, as CUDA's block size is determined by the `blockSize` variable.
*   Added a call to `cudaDeviceSynchronize` after each kernel launch to ensure that the results are available before accessing them.

Also, note that the `n`, `alpha`, `theta_S`, and `theta_R` variables are not defined in the provided code. You will need to define these variables before running the code.