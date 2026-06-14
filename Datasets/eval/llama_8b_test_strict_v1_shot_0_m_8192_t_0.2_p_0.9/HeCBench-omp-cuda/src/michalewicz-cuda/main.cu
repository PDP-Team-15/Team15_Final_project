#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>
#include <random>

#define min(a,b) (a) < (b) ? (a) : (b)

__device__ float michalewicz(const float *xValues, const int dim) {
  float result = 0;
  for (int i = 0; i < dim; ++i) {
      float a = sinf(xValues[i]);
      float b = sinf(((i + 1) * xValues[i] * xValues[i]) / (float)M_PI);
      float c = powf(b, 20); 

      result += a * c;
  }
  return -1.0f * result;
}

__global__ void Error(float value, int dim) {
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

int main(int argc, char* argv[]) {
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

    float *values;
    cudaMalloc((void**)&values, size * sizeof(float));
    
    float *d_values;
    cudaMalloc((void**)&d_values, size * sizeof(float));
    
    cudaMemcpy(d_values, values, size * sizeof(float), cudaMemcpyHostToDevice);

    float minValue = 0;
    
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      dim3 blocks(n / 256 + 1);
      dim3 threads(256);
      Error<<<blocks, threads>>>(minValue, dim);
      cudaDeviceSynchronize();
      minValue = 0;
      #pragma omp parallel for
      for (size_t j = 0; j < n; j++) {
        minValue = min(minValue, michalewicz(d_values + j * dim, dim));
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernel (dim = %d): %f (us)\n",
           dim, (time * 1e-3f) / repeat);
    cudaMemcpy(&minValue, d_values, sizeof(float), cudaMemcpyDeviceToHost);
    Error(minValue, dim);
    cudaFree(values);
    cudaFree(d_values);
  }

  return 0;
}