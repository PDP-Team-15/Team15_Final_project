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
void computeMin(float* values, float* minValue, const int dim, const size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float current = michalewicz(values + idx * dim, dim);
    if (current < *minValue) {
      *minValue = current;
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
    for (size_t i = 0; i < size; i++) {
      h_values[i] = dis(gen);
    }

    float* d_values;
    cudaMalloc((void**)&d_values, size_bytes);
    cudaMemcpy(d_values, h_values, size_bytes, cudaMemcpyHostToDevice);

    float* d_minValue;
    cudaMalloc((void**)&d_minValue, sizeof(float));
    cudaMemcpy(d_minValue, &h_values[0], sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    for (int i = 0; i < repeat; i++) {
      computeMin<<<blocks, threads>>>(d_values, d_minValue, dim, n);
    }

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    printf("Average execution time of kernel (dim = %d): %f (us)\n", dim, time / repeat);

    cudaMemcpy(&h_values[0], d_minValue, sizeof(float), cudaMemcpyDeviceToHost);
    Error(h_values[0], dim);

    free(h_values);
    cudaFree(d_values);
    cudaFree(d_minValue);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
  }

  return 0;
}