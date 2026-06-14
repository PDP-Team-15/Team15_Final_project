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

inline
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

void computeMinValue(float* values, int dim, int n, float* minValue, int threadsPerBlock) {
    int blockSize = threadsPerBlock;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    float* dev_values;
    float* dev_minValue;
    
    cudaMalloc((void**)&dev_values, n * dim * sizeof(float));
    cudaMalloc((void**)&dev_minValue, numBlocks * sizeof(float));
    
    cudaMemcpy(dev_values, values, n * dim * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 grid(numBlocks);
    dim3 block(blockSize);
    
    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < repeat; i++) {
        michalewiczKernel<<<grid, block>>>(dev_values, dim, n, dev_minValue);
        
        cudaMemcpy(minValue, dev_minValue, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
        
        float currentMin = minValue[0];
        for (int j = 1; j < numBlocks; j++) {
            currentMin = min(currentMin, minValue[j]);
        }
        minValue[0] = currentMin;
    }
    
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernel (dim = %d): %f (us)\n",
           dim, (time * 1e-3f) / repeat);
    
    cudaFree(dev_values);
    cudaFree(dev_minValue);
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

__global__ void michalewiczKernel(float* values, int dim, int n, float* minValue) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    extern __shared__ float sharedMin[];
    
    sharedMin[tid] = 0.0f;
    
    for (int j = bid * blockDim.x + tid; j < n; j += blockDim.x * gridDim.x) {
        float current = michalewicz(values + j * dim, dim);
        if (current < sharedMin[tid]) {
            sharedMin[tid] = current;
        }
    }
    
    __syncthreads();
    
    if (tid == 0) {
        float blockMin = sharedMin[0];
        for (int i = 1; i < blockDim.x; i++) {
            blockMin = min(blockMin, sharedMin[i]);
        }
        minValue[bid] = blockMin;
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
    
    float *values = (float*) malloc(size_bytes);
    
    for (size_t i = 0; i < size; i++) {
      values[i] = dis(gen);
    }

    float minValue = 0;
    float* minValueArray = &minValue;
    
    computeMinValue(values, dim, n, minValueArray, 256);
    
    Error(minValue, dim);
    free(values);
  }

  return 0;
}