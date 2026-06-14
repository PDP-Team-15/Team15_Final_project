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
#include <iostream>

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

__global__ void kernel(float *values, int dim, float *minValue) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < dim; i += stride) {
    float localMinValue = michalewicz(values + i, dim);
    if (localMinValue < *minValue) {
      *minValue = localMinValue;
    }
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
    
    float *values;
    cudaMalloc((void **)&values, size_bytes);
    
    for (size_t i = 0; i < size; i++) {
      float value;
      std::uniform_real_distribution<float> dis(0.0, 4.0);
      value = dis(gen);
      cudaMemcpy(&values[i], &value, sizeof(float), cudaMemcpyHostToDevice);
    }

    float minValue;
    cudaMemset(&minValue, 0, sizeof(float));
    
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      int blockSize = 256;
      int numBlocks = (n + blockSize - 1) / blockSize;
      kernel<<<numBlocks, blockSize>>>(values, dim, &minValue);
      cudaDeviceSynchronize();
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernel (dim = %d): %f (us)\n",
             dim, (time * 1e-3f) / repeat);
    cudaMemcpy(&minValue, &minValue, sizeof(float), cudaMemcpyDeviceToHost);
    Error(minValue, dim);
    cudaFree(values);
  }

  return 0;
}