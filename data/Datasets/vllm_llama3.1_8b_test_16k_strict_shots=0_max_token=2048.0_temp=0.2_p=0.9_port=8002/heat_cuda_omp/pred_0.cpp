```cpp
#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>
#include <omp.h>

#define PI acos(-1.0) 

#define LINE "--------------------" 


void initial_value(int n, double dx, double length, double * u);
void zero(int n, double * u);
void solve(int n, double alpha, double dx, double dt, double r, double r2,
		double * __restrict__ u, double * __restrict__ u_tmp);
double solution(double t, double x, double y, double alpha, double length);
double l2norm(int n, double * u, int nsteps, double dt, double alpha, double dx, double length);

int main(int argc, char *argv[]) {

  auto start = std::chrono::high_resolution_clock::now();

  int n = 1000;

  int nsteps = 10;

  if (argc == 3) {
    n = atoi(argv[1]);
    if (n < 0) {
      std::cerr << "Error: n must be positive" << std::endl;
      exit(EXIT_FAILURE);
    }
    nsteps = atoi(argv[2]);
    if (nsteps < 0) {
      std::cerr << "Error: nsteps must be positive" << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  double alpha = 0.1;           
  double length = 1000.0;      
  double dx = length / (n+1);  
  double dt = 0.5 / nsteps;    

  double r = alpha * dt / (dx * dx);
  double r2 = 1.0 - 4.0*r;

  std::cout
    << std::endl
    << " MMS heat equation" << std::endl << std::endl
    << LINE << std::endl
    << "Problem input" << std::endl << std::endl
    << " Grid size: " << n << " x " << n << std::endl
    << " Cell width: " << dx << std::endl
    << " Grid length: " << length << "x" << length << std::endl
    << std::endl
    << " Alpha: " << alpha << std::endl
    << std::endl
    << " Steps: " <<  nsteps << std::endl
    << " Total time: " << dt*(double)nsteps << std::endl
    << " Time step: " << dt << std::endl
    << LINE << std::endl;

  std::cout << "Stability" << std::endl << std::endl;
  std::cout << " r value: " << r << std::endl;
  if (r > 0.5)
    std::cout << " Warning: unstable" << std::endl;
  std::cout << LINE << std::endl;

  double *u;
  double *u_tmp;
  u = new double[n*n];
  u_tmp = new double[n*n];

  initial_value(n, dx, length, u);
  zero(n, u_tmp);

  double tic = omp_get_wtime();

  #pragma omp parallel for
  for (int t = 0; t < nsteps; ++t) {
    solve(n, alpha, dx, dt, r, r2, u, u_tmp);
    double *tmp = u;
    u = u_tmp;
    u_tmp = tmp;
  }

  double toc = omp_get_wtime();

  double *u_host = new double[n*n];
  for (int i = 0; i < n*n; ++i) {
    u_host[i] = u[i];
  }

  double norm = l2norm(n, u_host, nsteps, dt, alpha, dx, length);

  std::cout
    << "Results" << std::endl << std::endl
    << "Error (L2norm): " << norm << std::endl
    << "Solve time (s): " << toc-tic << std::endl
    << "Total time (s): " << omp_get_wtime()-start << std::endl
    << LINE << std::endl;

  delete[] u_host;
  delete[] u;
  delete[] u_tmp;

  return 0;
}

void initial_value(int n, double dx, double length, double * u) {
  #pragma omp parallel for
  for (int idx = 0; idx < n*n; ++idx) {
    int i = idx % n;
    int j = idx / n;
    double y = dx * (j+1); 
    double x = dx * (i+1); 
    u[idx] = sin(PI * x / length) * sin(PI * y / length);
  }
}

void zero(int n, double * u) {
  #pragma omp parallel for
  for (int idx = 0; idx < n*n; ++idx) {
    u[idx] = 0.0;
  }
}

void solve(int n, double alpha, double dx, double dt, double r, double r2,
		double * __restrict__ u, double * __restrict__ u_tmp) {
  #pragma omp parallel for
  for (int idx = 0; idx < n*n; ++idx) {
    int i = idx % n;
    int j = idx / n;
    u_tmp[idx] =  r2 * u[idx] +
    r * ((i < n-1) ? u[idx+1] : 0.0) +
    r * ((i > 0)   ? u[idx-1] : 0.0) +
    r * ((j < n-1) ? u[idx+n] : 0.0) +
    r * ((j > 0)   ? u[idx-n] : 0.0);
  }
}

double solution(double t, double x, double y, double alpha, double length) {
  return exp(-2.0*alpha*PI*PI*t/(length*length)) * sin(PI*x/length) * sin(PI*y/length);
}

double l2norm(int n, double * u, int nsteps, double dt, double alpha, double dx, double length) {
  double time = dt * (double)nsteps;
  double l2norm = 0.0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      double answer = solution(time, dx*i, dx*j, alpha, length);
      l2norm += (u[i+j*n] - answer) * (u[i+j*n] - answer);
    }
  }
  return sqrt(l2norm);
}
```