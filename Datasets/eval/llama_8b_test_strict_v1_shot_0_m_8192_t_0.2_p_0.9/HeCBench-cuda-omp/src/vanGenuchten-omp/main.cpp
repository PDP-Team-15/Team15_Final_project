#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
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

  #pragma omp parallel for default(none) \
  shared(Ksat, psi, C, theta, K, size) \
  private(i, lambda, m, _psi, _theta, Se, t)
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

  #pragma omp parallel for default(none) \
  shared(Ksat, psi, C, theta, K, size) \
  private(i)
  for (i = 0; i < size; i++) {
    Ksat[i] = Ksat[i];
    psi[i] = psi[i];
  }

  double *Ksat_omp, *psi_omp, *C_omp, *theta_omp, *K_omp;
  Ksat_omp = (double*)malloc(size_byte);
  psi_omp = (double*)malloc(size_byte);
  C_omp = (double*)malloc(size_byte);
  theta_omp = (double*)malloc(size_byte);
  K_omp = (double*)malloc(size_byte);

  for (int i = 0; i < size; i++) {
    Ksat_omp[i] = Ksat[i];
    psi_omp[i] = psi[i];
  }

  double start = omp_get_wtime();

  for (int i = 0; i < repeat; i++)
    vanGenuchten(Ksat_omp, psi_omp, C_omp, theta_omp, K_omp, size);

  double end = omp_get_wtime();
  double time = (end - start) / repeat;
  printf("Average kernel execution time: %f (s)\n", time);

  for (int i = 0; i < size; i++) {
    C[i] = C_omp[i];
    theta[i] = theta_omp[i];
    K[i] = K_omp[i];
  }

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

  free(Ksat_omp);
  free(psi_omp);
  free(C_omp);
  free(theta_omp);
  free(K_omp);

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