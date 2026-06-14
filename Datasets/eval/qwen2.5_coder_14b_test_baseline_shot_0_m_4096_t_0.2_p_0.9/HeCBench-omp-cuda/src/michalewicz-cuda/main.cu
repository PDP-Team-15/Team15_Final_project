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
float michalewicz_device(const float *xValues, const int dim) {
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
void michalewicz_kernel(const float *values, float *minValue, const int dim, const int n) {
  extern __shared__ float shared_min[];
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  float local_min = michalewicz_device(values + idx * dim, dim);

  // Reduce within block
  int tid = threadIdx.x;
  shared_min[tid] = local_min;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_min[tid] = min(shared_min[tid], shared_min[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicMin(minValue, shared_min[0]);
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

    float minValue = 0;
    float *d_values, *d_minValue;
    cudaMalloc(&d_values, size_bytes);
    cudaMemcpy(d_values, values, size_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_minValue, sizeof(float));

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      cudaMemcpy(d_minValue, &minValue, sizeof(float), cudaMemcpyHostToDevice);
      int threadsPerBlock = 256;
      int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
      size_t sharedMemSize = threadsPerBlock * sizeof(float);
      michalewicz_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_values, d_minValue, dim, n);
      cudaMemcpy(&minValue, d_minValue, sizeof(float), cudaMemcpyDeviceToHost);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernel (dim = %d): %f (us)\n",
           dim, (time * 1e-3f) / repeat);

    Error(minValue, dim);

    free(values);
    cudaFree(d_values);
    cudaFree(d_minValue);
  }

  return 0;
}