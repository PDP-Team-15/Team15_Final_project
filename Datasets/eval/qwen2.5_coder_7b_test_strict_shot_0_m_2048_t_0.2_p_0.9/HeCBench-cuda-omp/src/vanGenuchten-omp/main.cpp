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
  const double *Ksat,
  const double *psi,
        double *C,
        double *theta,
        double *K,
  const int size)
{
  double Se, _theta, _psi, lambda, m, t;

  #pragma omp parallel for private(lambda, m, _psi, _theta, Se, t)
  for (int i = 0; i < size; i++)
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

  double *d_Ksat, *d_psi, *d_C, *d_theta, *d_K;
  // Simulating device memory allocation and copying
  d_Ksat = Ksat;
  d_psi = psi;
  d_C = C;
  d_theta = theta;
  d_K = K;

  double start_time = omp_get_wtime();

  for (int i = 0; i < repeat; i++)
    vanGenuchten(d_Ksat, d_psi, d_C, d_theta, d_K, size);

  double end_time = omp_get_wtime();
  double time = (end_time - start_time) / repeat;
  printf("Average kernel execution time: %f (s)\n", time);

  // Simulating device to host data transfer
  memcpy(C, d_C, size_byte);
  memcpy(theta, d_theta, size_byte);
  memcpy(K, d_K, size_byte);

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