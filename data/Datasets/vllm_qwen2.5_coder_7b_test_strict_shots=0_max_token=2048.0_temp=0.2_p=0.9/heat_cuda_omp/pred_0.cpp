```cpp
#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>

#define PI acos(-1.0) 
#define LINE "--------------------" 

#pragma omp declare target
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

  initial_value(n, dx, length, u);
  zero(n, u_tmp);

  const double r2 = 1.0 - 4.0*r;

  auto tic = std::chrono::high_resolution_clock::now();

  #pragma omp parallel for collapse(2)
  for (int t = 0; t < nsteps; ++t) {
    solve(n, alpha, dx, dt, r, r2, u, u_tmp);
    double *tmp = u;
    u = u_tmp;
    u_tmp = tmp;
  }

  auto toc = std::chrono::high_resolution_clock::now();

  double *u_host = new double[n*n];
  memcpy(u_host, u, sizeof(double)*n*n);

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
}

void initial_value(const unsigned int n, const double dx, const double length, double * u) {
  for (unsigned int idx = 0; idx < n*n; ++idx) {
    int i = idx % n;
    int j = idx / n;
    double y = dx * (j+1); 
    double x = dx * (i+1); 
    u[idx] = sin(PI * x / length) * sin(PI * y / length);
  }
}

void zero(const unsigned int n, double * u) {
  for (unsigned int idx = 0; idx < n*n; ++idx) u[idx] = 0.0;
}

void solve(const unsigned int n, const double alpha, const double dx, const double dt, 
         const double r, const double r2,
         double * __restrict__ u, double * __restrict__ u_tmp) {
  for (unsigned int idx = 0; idx < n * n; ++idx) {
    int i = idx % n;
    int j = idx / n;
    u_tmp[idx] =  r2 * u[idx] +
                  r * ((i < n-1) ? u[(i+1)+j*n] : 0.0) +
                  r * ((i > 0)   ? u[(i-1)+j*n] : 0.0) +
                  r * ((j < n-1) ? u[i+(j+1)*n] : 0.0) +
                  r * ((j > 0)   ? u[i+(j-1)*n] : 0.0);
  }
}

double solution(const double t, const double x, const double y, const double alpha, const double length) {
  return exp(-2.0*alpha*PI*PI*t/(length*length)) * sin(PI*x/length) * sin(PI*y/length);
}

double l2norm(const int n, const double * u, const int nsteps, const double dt, const double alpha, const double dx, const double length) {
  double time = dt * (double)nsteps;
  double l2norm = 0.0;
  double y = dx;
  for (int j = 0; j < n; ++j) {
    double x = dx;
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