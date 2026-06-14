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
__global__ void kernel(const T* data, T* gpuData, const int len) {
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
__global__ void max_kernel(T* gpuData, const int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    gpuData[2] = max(gpuData[2], (T)idx);
  }
}

template <class T>
__global__ void min_kernel(T* gpuData, const int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    gpuData[3] = min(gpuData[3], (T)idx);
  }
}

template <class T>
void testcase(const int repeat) {
  const int len = 1 << 10;
  const int numThreads = 256;
  const int numData = 7;
  const int memSize = sizeof(T) * numData;
  T data[] = {0, 0, (T)-256, 256, 255, 0, 255};
  T* gpuData;
  CUDA_CHECK(cudaMalloc((void**)&gpuData, memSize));
  CUDA_CHECK(cudaMemcpy(gpuData, data, memSize, cudaMemcpyHostToDevice));

  for (int n = 0; n < repeat; n++) {
    CUDA_KERNEL_LOOP(i, len);
    kernel<<<dim3((len + 255 - 1) / 256), 256>>>(data, gpuData, len);
    CUDA_CHECK(cudaDeviceSynchronize());

    max_kernel<<<dim3((len + 255 - 1) / 256), 256>>>(gpuData, len);
    CUDA_CHECK(cudaDeviceSynchronize());

    min_kernel<<<dim3((len + 255 - 1) / 256), 256>>>(gpuData, len);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_KERNEL_LOOP(i, len);
    kernel<<<dim3((len + 255 - 1) / 256), 256>>>(data, gpuData, len);
    CUDA_CHECK(cudaDeviceSynchronize());

    max_kernel<<<dim3((len + 255 - 1) / 256), 256>>>(gpuData, len);
    CUDA_CHECK(cudaDeviceSynchronize());

    min_kernel<<<dim3((len + 255 - 1) / 256), 256>>>(gpuData, len);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  CUDA_CHECK(cudaMemcpy(data, gpuData, memSize, cudaMemcpyDeviceToHost));
  computeGold<T>(data, len);

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    CUDA_KERNEL_LOOP(i, len);
    kernel<<<dim3((len + 255 - 1) / 256), 256>>>(data, gpuData, len);
    CUDA_CHECK(cudaDeviceSynchronize());

    max_kernel<<<dim3((len + 255 - 1) / 256), 256>>>(gpuData, len);
    CUDA_CHECK(cudaDeviceSynchronize());

    min_kernel<<<dim3((len + 255 - 1) / 256), 256>>>(gpuData, len);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  CUDA_CHECK(cudaFree(gpuData));
}

int main(int argc, char** argv) {
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

Note that I've assumed the `computeGold` function is defined in the `reference.h` header file and is not modified in the CUDA translation. If this is not the case, you'll need to modify the CUDA code accordingly.