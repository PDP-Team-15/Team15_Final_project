```cpp
#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>
#include <omp.h>

#define PI acos(-1.0) 
#define LINE "--------------------" 

void initial_value(const unsigned int n, const double dx, const double length, double * u);
void zero(const unsigned int n, double * u);
void solve(const unsigned int n, const double alpha, const double dx, const double dt, const double r, const double r2,
		double * __restrict__ u, double * __restrict__ u_tmp);
double solution(const double t, const double x, const double y, const double alpha, const double length);
double l2norm(const int n, const double * u, const int nsteps, const double dt, const double alpha, const double dx, const double length);

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

  double *u = new double[n*n];
  double *u_tmp = new double[n*n];

  int num_threads = omp_get_max_threads();
  int n_ceil = (n*n+num_threads-1) / num_threads;

  #pragma omp parallel for
  for (int i = 0; i < n*n; ++i) {
    int idx = i;
    int i_ = idx % n;
    int j_ = idx / n;
    double y = dx * (j_+1); 
    double x = dx * (i_+1); 
    u[i] = sin(PI * x / length) * sin(PI * y / length);
  }

  #pragma omp parallel for
  for (int i = 0; i < n*n; ++i) {
    u_tmp[i] = 0.0;
  }

  double tic = omp_get_wtime();

  for (int t = 0; t < nsteps; ++t) {

    #pragma omp parallel for
    for (int i = 0; i < n*n; ++i) {
      int idx = i;
      int i_ = idx % n;
      int j_ = idx / n;
      double u_tmp_i =  r2 * u[i] +
      r * ((i_ < n-1) ? u[i+1] : 0.0) +
      r * ((i_ > 0)   ? u[i-1] : 0.0) +
      r * ((j_ < n-1) ? u[i+(j_+1)*n] : 0.0) +
      r * ((j_ > 0)   ? u[i+(j_-1)*n] : 0.0);
      u_tmp[i