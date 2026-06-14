#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <omp.h>

#pragma omp declare reduction(minimum : float : omp_out = fminf(omp_out, omp_in)) initializer(omp_priv = INFINITY)

float atomic_min(float *addr, float value)
{
  #pragma omp critical
  {
    if (value < *addr) {
      *addr = value;
    }
  }
  return *addr;
}

float michalewicz(const float *xValues, const int dim) {
  float result = 0;
  for (int i = 0; i < dim; ++i) {
    float a = sinf(xValues[i]);
    float b = sinf(((i + 1) * xValues[i] * xValues[i]) / (float)M_PI);
    float c = powf(b, 20); 

    result += a * c;
  }
  return -1.0f * result;
}

void eval (const float *values, float *minima,
          const size_t nVectors, const int dim)
{
  #pragma omp parallel for reduction(minimum : minima)
  for (size_t n = 0; n < nVectors; ++n) {
    atomic_min(minima, michalewicz(values + n * dim, dim));
  }
}

void Error(float value, int dim) {
  printf("Global minima = %f\n", value);
  float trueMin = 0.0;
  if (dim == 2)
    trueMin = -1.8013;
  else if (dim == 5)
    trueMin = -4.687658;
  else if (dim == 10)
    trueMin = -9.66015;
  printf("Error = %f\n", fabsf(trueMin - value));
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of vectors> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t n = atol(argv[1]);
  const int repeat = atoi(argv[2]);

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dis(0.0, 4.0);
  
  const int dims[] = {2, 5, 10}; 

  for (int d = 0; d < 3; d++) {

    const int dim = dims[d];

    const size_t size = n * dim;

    const size_t size_bytes = size * sizeof(float);
    
    float *values = (float*) malloc (size_bytes);
    
    for (size_t i = 0; i < size; i++) {
      values[i] = dis(gen);
    }
    
    float minValue = INFINITY;

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      eval(values, &minValue, n, dim);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernel (dim = %d): %f (us)\n",
           dim, (time * 1e-3f) / repeat);

    Error(minValue, dim);

    free(values);
  }

  return 0;
}