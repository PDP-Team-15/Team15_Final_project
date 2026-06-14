Here is the CUDA translation of the provided OpenMP code:

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
    Ksat[i] = 1e-6 + (1.0 - 1e-6) * i / size;
    psi[i] = -100.0 + 101.0 * i / size;
  }

  reference(Ksat, psi, C_ref, theta_ref, K_ref, size);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    vanGenuchten(Ksat, psi, C, theta, K, size);

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
```

### Explanation:
1. **CUDA Kernel Function (`vanGenuchtenKernel`)**:
   - The kernel function is defined with `__global__` and executed on the GPU.
   - It calculates the same values as the original OpenMP loop but in a parallel manner using threads.

2. **Host Function (`vanGenuchten`)**:
   - This function calls the CUDA kernel and synchronizes the device to ensure all operations are completed before proceeding.

3. **Memory Management**:
   - Memory is allocated using `cudaMallocManaged`, which allows for unified memory accessible by both CPU and GPU.

4. **Timing**:
   - Timing is done similarly to the original code using `std::chrono`.

5. **Main Function**:
   - The main function remains unchanged except for the memory allocation and deallocation.

This translation ensures that the computation is performed on the GPU, leveraging parallel processing capabilities.