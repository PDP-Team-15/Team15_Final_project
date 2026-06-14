```cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <random>

float atomic_min(float *addr, float value)
{
  unsigned ret = *(unsigned *)addr;
  while(value < __uint_as_float(ret))
  {
    unsigned old = ret;
    if((ret = __atomic_compare_exchange_n(addr, &old, __float_as_uint(value), false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) == old)
      break;
  }
  return __uint_as_float(ret);
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

void eval(float *values, float *minima,
          const size_t nVectors, const int dim)
{
  #pragma omp parallel for reduction(min:minima[0])
  for (size_t n = 0; n < nVectors; ++n) {
    minima[0] = atomic_min(minima, michalewicz(values + n * dim, dim));
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

    double start = omp_get_wtime();

    for (int i = 0; i < repeat; i++)
      eval(values, &minValue, n, dim);

    double end = omp_get_wtime();
    printf("Average execution time of kernel (dim = %d): %f (us)\n",
           dim, (end - start) * 1e6 / repeat);

    Error(minValue, dim);

    free(values);
  }

  return 0;
}
```