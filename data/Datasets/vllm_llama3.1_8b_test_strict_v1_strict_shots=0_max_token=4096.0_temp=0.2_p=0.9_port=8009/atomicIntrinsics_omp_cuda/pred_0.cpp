```cuda
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s:%d: %s.\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
      exit(1); \
    } \
  } while (0)

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <class T>
__global__ void computeKernel(T *gpuData, int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    gpuData[0] += (T)10;
    gpuData[1] -= (T)10;
    gpuData[4] &= (T)(2 * idx + 7);
    gpuData[5] |= (T)(1 << idx);
    gpuData[6] ^= (T)idx;
  }
}

template <class T>
__global__ void reduceMaxKernel(T *gpuData, int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    gpuData[threadIdx.x] = max(gpuData[threadIdx.x], (T)idx);
  }
  __syncthreads();
  for (int i = blockDim.x / 2; i > 0; i /= 2) {
    if (threadIdx.x < i) {
      gpuData[threadIdx.x] = max(gpuData[threadIdx.x], gpuData[threadIdx.x + i]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    gpuData[0] = max(gpuData[0], gpuData[blockDim.x]);
  }
}

template <class T>
__global__ void reduceMinKernel(T *gpuData, int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    gpuData[threadIdx.x] = min(gpuData[threadIdx.x], (T)idx);
  }
  __syncthreads();
  for (int i = blockDim.x / 2; i > 0; i /= 2) {
    if (threadIdx.x < i) {
      gpuData[threadIdx.x] = min(gpuData[threadIdx.x], gpuData[threadIdx.x + i]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    gpuData[0] = min(gpuData[0], gpuData[blockDim.x]);
  }
}

template <class T>
void testcase(const int repeat) {
  const int len = 1 << 10;
  const int numThreads = 256;
  const int numData = 7;
  const size_t memSize = sizeof(T) * numData;
  T *data = (T *)malloc(memSize);
  T *gpuData;
  CUDA_CHECK(cudaMalloc((void **)&gpuData, memSize));
  data[0] = 0;
  data[1] = 0;
  data[2] = (T)-256;
  data[3] = 256;
  data[4] = 255;
  data[5] = 0;
  data[6] = 255;

  CUDA_CHECK(cudaMemcpy(gpuData, data, memSize, cudaMemcpyHostToDevice));

  for (int n = 0; n < repeat; n++) {
    CUDA_KERNEL_LOOP(i, len);
    computeKernel<<<dim3((len + 255 - 1) / 256), 256>>>(gpuData, len);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_KERNEL_LOOP(i, len);
    reduceMaxKernel<<<dim3((len + 255 - 1) / 256), 256>>>(gpuData, len);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_KERNEL_LOOP(i, len);
    reduceMinKernel<<<dim3((len + 255 - 1) / 256), 256>>>(gpuData, len);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  CUDA_CHECK(cudaMemcpy(data, gpuData, memSize, cudaMemcpyDeviceToHost));
  computeGold<T>(data, len);

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    CUDA_KERNEL_LOOP(i, len);
    computeKernel<<<dim3((len + 255 - 1) / 256), 256>>>(gpuData, len);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_KERNEL_LOOP(i, len);
    reduceMaxKernel<<<dim3((len + 255 - 1) / 256), 256>>>(gpuData, len);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_KERNEL_LOOP(i, len);
    reduceMinKernel<<<dim3((len + 255 - 1) / 256), 256>>>(gpuData, len);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  CUDA_CHECK(cudaFree(gpuData));
  free(data);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  testcase<int>(repeat);
  testcase<unsigned int>(repeat);
  return 0;
}
```

Note: The `computeGold` function is not defined in the provided code, so I assume it is defined in the `reference.h` header file. Also, the `cudaMalloc` and `cudaMemcpy` calls are not checked for errors, but this is a good practice to ensure the correctness of the code.