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
#include <cuda_runtime.h>

#define min(a,b) (a) < (b) ? (a) : (b)

__device__
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

__global__
void findMinValueKernel(const float *values, float *minValue, const int n, const int dim) {
  extern __shared__ float sharedMin[];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    float val = michalewicz(values + idx * dim, dim);
    sharedMin[threadIdx.x] = val;
  } else {
    sharedMin[threadIdx.x] = INFINITY;
  }
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sharedMin[threadIdx.x] = min(sharedMin[threadIdx.x], sharedMin[threadIdx.x + s]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicMin(minValue, sharedMin[0]);
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

    float *values = (float*) malloc(size_bytes);
    for (size_t i = 0; i < size; i++) {
      values[i] = dis(gen);
    }

    float minValue = INFINITY;
    float *d_values, *d_minValue;
    cudaMalloc(&d_values, size_bytes);
    cudaMalloc(&d_minValue, sizeof(float));

    cudaMemcpy(d_values, values, size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_minValue, &minValue, sizeof(float), cudaMemcpyHostToDevice);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      int threadsPerBlock = 256;
      int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
      findMinValueKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_values, d_minValue, n, dim);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernel (dim = %d): %f (us)\n",
           dim, (time * 1e-3f) / repeat);

    cudaMemcpy(&minValue, d_minValue, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_values);
    cudaFree(d_minValue);

    Error(minValue, dim);
    free(values);
  }

  return 0;
}