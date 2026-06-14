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
void solve(int n, double alpha, double dx, double dt, double r, double r2, double * __restrict__ u, double * __restrict__ u_tmp);
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

  int num_threads = omp_get_max_threads();
  int n_ceil = (n*n+num_threads-1) / num_threads;
  #pragma omp parallel for
  for (int i = 0; i < n*n; ++i) {
    int idx = i;
    int i_ = idx % n;
    int j_ = idx / n;
    double y = dx * (j_+1); 

    double x = dx * (i_+1); 

    u[idx] = sin(PI * x / length) * sin(PI * y / length);
  }

  #pragma omp parallel for
  for (int i = 0; i < n*n; ++i) {
    u_tmp[i] = 0.0;
  }

  double r2 = 1.0 - 4.0*r;

  auto tic = std::chrono::high_resolution_clock::now();

  for (int t = 0; t < nsteps; ++t) {

    #pragma omp parallel for
    for (int i = 0; i < n*n; ++i) {
      int idx = i;
      int i_ = idx % n;
      int j_ = idx / n;

      u_tmp[idx] =  r2 * u[idx] +
      r * ((i_ < n-1) ? u[i_+1+j_*n] : 0.0) +
      r * ((i_ > 0)   ? u[i_-1+j_*n] : 0.0) +
      r * ((j_ < n-1) ? u[i_+(j_+1)*n] : 0.0) +
      r * ((j_ > 0)   ? u[i_+(j_-1)*n] : 0.0);
    }

    double *tmp = u;
    u = u_tmp;
    u_tmp = tmp;
  }

  auto toc = std::chrono::high_resolution_clock::now();

  double *u_host = new double[n*n];
  #pragma omp parallel for
  for (int i = 0; i < n*n; ++i) {
    u_host[i] = u[i];
  }

  double norm = l2norm(n, u_host, nsteps, dt, alpha, dx, length);

  auto stop = std::chrono::high_resolution_clock::now();

  std::cout
    << "Results" << std::endl << std::endl
    << "Error (L2norm): " << norm << std::endl
    << "Solve time (s): " << std::chrono::duration_cast<std::chrono::duration<double>>(toc-tic).count() << std::endl
    << "Total time (s): " << std::chrono::duration_cast<std::chrono::duration<double>>(stop-start).count() << std::endl
    << "Bandwidth (GB/s): " << 1.0E-9*2.0*n*n*nsteps*sizeof(double)/std::chrono::duration_cast<std::chrono::duration<double>>(toc-tic).count() << std::endl
    << LINE << std::endl;

  delete[] u_host;
  delete[] u;
  delete[] u_tmp;

  return 0;
}

void initial_value(int n, double dx, double length, double * u) {
  int num_threads = omp_get_max_threads();
  int n_ceil = (n*n+num_threads-1) / num_threads;
  #pragma omp parallel for
  for (int i = 0; i < n*n; ++i) {
    int idx = i;
    int i_ = idx % n;
    int j_ = idx / n;
    double y = dx * (j_+1); 

    double x = dx * (i_+1); 

    u[idx] = sin(PI * x / length) * sin(PI * y / length);
  }
}

void zero(int n, double * u) {
  int num_threads = omp_get_max_threads();
  int n_ceil = (n*n+num_threads-1) / num_threads;
  #pragma omp parallel for
  for (int i = 0; i < n*n; ++i) {
    u[i] = 0.0;
  }
}

void solve(int n, double alpha, double dx, double dt, double r, double r2, double * __restrict__ u, double * __restrict__ u_tmp) {
  int num_threads = omp_get_max_threads();
  int n_ceil = (n*n+num_threads-1) / num_threads;
  #pragma omp parallel for
  for (int i = 0; i < n*n; ++i) {
    int idx = i;
    int i_ = idx % n;
    int j_ = idx / n;

    u_tmp[idx] =  r2 * u[idx] +
    r * ((i_ < n-1) ? u[i_+1+j_*n] : 0.0) +
    r * ((i_ > 0)   ? u[i_-1+j_*n] : 0.0) +
    r * ((j_ < n-1) ? u[i_+(j_+1)*n] : 0.0) +
    r * ((j_ > 0)   ? u[i_+(j_-1)*n] : 0.0);
  }
}

double solution(double t, double x, double y, double alpha, double length) {
  return exp(-2.0*alpha*PI*PI*t/(length*length)) * sin(PI*x/length) * sin(PI*y/length);
}

double l2norm(int n, double * u, int nsteps, double dt, double alpha, double dx, double length) {
  double time = dt * (double)nsteps;

  double l2norm = 0.0;

  double y = dx;
  #pragma omp parallel for
  for (int j = 0; j < n; ++j) {
    double x = dx;
    #pragma omp parallel for reduction(+:l2norm)
    for (int i = 0; i < n; ++i) {
      double answer = solution(time, x, y, alpha, length);
      l2norm += (u[i+j*n] - answer) * (u[i+j*n] - answer);

      x += dx;
    }
    y += dx;
  }

  return sqrt(l2norm);
}
```