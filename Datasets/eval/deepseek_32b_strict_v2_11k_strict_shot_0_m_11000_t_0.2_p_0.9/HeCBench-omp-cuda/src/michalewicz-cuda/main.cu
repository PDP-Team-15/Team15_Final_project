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

__global__
void compute_min_value(float* values, int n, int dim, float* minValue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float current = michalewicz(values + idx * dim, dim);
        if (current < *minValue) {
            atomicMin(minValue, current);
        }
    }
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

    float* h_values = (float*)malloc(size_bytes);
    float* d_values;
    cudaMalloc(&d_values, size_bytes);

    for (size_t i = 0; i < size; i++) {
      h_values[i] = dis(gen);
    }

    cudaMemcpy(d_values, h_values, size_bytes, cudaMemcpyHostToDevice);

    float minValue = 0;
    float* d_minValue;
    cudaMalloc(&d_minValue, sizeof(float));
    cudaMemcpy(d_minValue, &minValue, sizeof(float), cudaMemcpyHostToDevice);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        compute_min_value<<<blocksPerGrid, threadsPerBlock>>>(d_values, n, dim, d_minValue);
    }

    cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernel (dim = %d): %f (us)\n",
           dim, (time * 1e-3f) / repeat);

    cudaMemcpy(&minValue, d_minValue, sizeof(float), cudaMemcpyDeviceToHost);
    Error(minValue, dim);

    free(h_values);
    cudaFree(d_values);
    cudaFree(d_minValue);
  }

  return 0;
}