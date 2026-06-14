```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "reference.h"

void vanGenuchten(
  const double *__restrict__ Ksat,
  const double *__restrict__ psi,
        double *__restrict__ C,
        double *__restrict__ theta,
        double *__restrict__ K,
  const int size)
{
  double Se, _theta, _psi, lambda, m, t;

  int i;
  #pragma omp parallel for private(i, lambda, m, _psi, _theta, Se, t) 
  for (i = 0; i < size; i++)
  {
    lambda = n - 1.0;
    m = lambda/n;

    _psi = psi[i] * 100.0;
    if ( _psi < 0.0 )
      _theta = (theta_S - theta_R) / pow(1.0 + pow((alpha*(-_psi)),n), m) + theta_R;
    else
      _theta = theta_S;

    theta[i] = _theta;

    Se = (_theta - theta_R)/(theta_S - theta_R);

    t = 1.0 - pow(1.0-pow(Se,1.0/m), m);
    K[i] = Ksat[i] * sqrt(Se) * t * t;

    if (_psi < 0.0)
      C[i] = 100.0 * alpha * n * (1.0/n-1.0)*pow(alpha*abs(_psi), n-1.0)
        * (theta_R-theta_S) * pow(pow(alpha*abs(_psi), n)+1.0, 1.0/n-2.0);
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
  const int size_byte = size * sizeof(double);

  double *Ksat, *psi, *C, *theta, *K;
  double *C_ref, *theta_ref, *K_ref;
  
  Ksat = (double*)malloc(size_byte);
  psi = (double*)malloc(size_byte);
  C = (double*)malloc(size_byte);
  theta = (double*)malloc(size_byte);
  K = (double*)malloc(size_byte);

  C_ref = (double*)malloc(size_byte);
  theta_ref = (double*)malloc(size_byte);
  K_ref = (double*)malloc(size_byte);

  for (int i = 0; i < size; i++) {
    Ksat[i] = 1e-6 +  (1.0 - 1e-6) * i / size; 
    psi[i] = -100.0 + 101.0 * i / size;
  }

  reference(Ksat, psi, C_ref, theta_ref, K_ref, size);

  #pragma omp parallel for
  for (int i = 0; i < repeat; i++)
    vanGenuchten(Ksat, psi, C, theta, K, size);

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

  free(Ksat);
  free(psi);
  free(C);
  free(theta);
  free(K);
  free(C_ref);
  free(theta_ref);
  free(K_ref);

  return 0;
}
```

Note that I've removed the CUDA-specific code and replaced it with OpenMP directives. I've also removed the `cudaMalloc` and `cudaFree` calls, as they are not needed in the OpenMP version. Additionally, I've removed the `cudaMemcpy` calls, as they are not needed in the OpenMP version. Instead, I've used OpenMP's `parallel for` directive to parallelize the computation.