```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <chrono>
#include <random>

float atomic_min(float *addr, float value)
{
  unsigned ret = *(unsigned*)addr;
  while(value < ret)
  {
    unsigned old = ret;
    if((ret = __sync_val_compare_and_swap((unsigned*)addr, old, (unsigned)value)) == old)
      break;
  }
  return (float)ret;
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
  size_t n = omp_get_thread_num();
  if (n < nVectors) {
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

    float *values = (float*) malloc (size * sizeof(float));
    
    for (size_t i = 0; i < size; i++) {
      values[i] = dis(gen);
    }
    
    float *minValue;
    float *minValueThread;
    minValue = (float*) malloc(sizeof(float));
    minValueThread = (float*) malloc(sizeof(float));

    #pragma omp parallel for
    for (int i = 0; i < repeat; i++) {
      float *valuesThread = (float*) malloc(size * sizeof(float));
      for (size_t j = 0; j < size; j++) {
        valuesThread[j] = dis(gen);
      }
      eval(valuesThread, minValueThread, n, dim);
      free(valuesThread);
    }

    #pragma omp critical
    {
      *minValue = atomic_min(minValue, *minValueThread);
    }

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      eval(values, minValue, n, dim);
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernel (dim = %d): %f (us)\n",
           dim, (time * 1e-3f) / repeat);

    Error(*minValue, dim);

    free(values);
    free(minValue);
    free(minValueThread);
  }

  return 0;
}
```
Note that I've used OpenMP's `#pragma omp parallel for` directive to parallelize the kernel launch, and `#pragma omp critical` to ensure that the atomic minimum update is thread-safe. I've also used `omp_get_thread_num()` to get the thread ID, which is equivalent to `blockIdx.x * blockDim.x + threadIdx.x` in CUDA.