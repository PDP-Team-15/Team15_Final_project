```cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>
#include <random>

#define min(a,b) (a) < (b) ? (a) : (b)

__global__ void michalewiczKernel(float *xValues, int dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dim) {
    float a = sinf(xValues[idx]);
    float b = sinf(((idx + 1) * xValues[idx] * xValues[idx]) / (float)M_PI);
    float c = powf(b, 20); 

    __shared__ float minValue;
    if (threadIdx.x == 0) {
      minValue = -1.0f * a * c;
    }
    __syncthreads();
    minValue = min(minValue, minValue);
    if (threadIdx.x == 0) {
      xValues[idx] = minValue;
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
    cudaMalloc((void**)&values, size_bytes);
    
    for (size_t i = 0; i < size; i++) {
      values[i] = dis(gen);
    }

    float minValue = 0;
    
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      michalewiczKernel<<<(dim + 255) / 256, 256>>>(values, dim);
      cudaDeviceSynchronize();
      minValue = -1.0f * values[0];
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernel (dim = %d): %f (us)\n",
           dim, (time * 1e-3f) / repeat);
    cudaFree(values);
  }

  return 0;
}
```