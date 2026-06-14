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

__global__ void find_min_value(float *values, float *minValue, int n, int dim) {
  __shared__ float shared_minValue[256];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  float local_minValue = INFINITY;
  while (i < n) {
    local_minValue = min(local_minValue, michalewicz(values + i * dim, dim));
    i += blockDim.x * gridDim.x;
  }

  shared_minValue[tid] = local_minValue;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_minValue[tid] = min(shared_minValue[tid], shared_minValue[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicMin(minValue, shared_minValue[0]);
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
    const size_t size_bytes = size * sizeof(float);

    float *h_values = (float*) malloc(size_bytes);
    float *d_values, *d_minValue;

    cudaMalloc((void**)&d_values, size_bytes);
    cudaMalloc((void**)&d_minValue, sizeof(float));

    for (size_t i = 0; i < size; i++) {
      h_values[i] = dis(gen);
    }

    cudaMemcpy(d_values, h_values, size_bytes, cudaMemcpyHostToDevice);
    *d_minValue = INFINITY;

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      find_min_value<<<(n + 255) / 256, 256>>>(d_values, d_minValue, n, dim);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernel (dim = %d): %f (us)\n",
           dim, (time * 1e-3f) / repeat);

    float minValue;
    cudaMemcpy(&minValue, d_minValue, sizeof(float), cudaMemcpyDeviceToHost);
    Error(minValue, dim);

    free(h_values);
    cudaFree(d_values);
    cudaFree(d_minValue);
  }

  return 0;
}