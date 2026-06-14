```cuda
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>
#include "reference.h"

#define CUDA_CHECK(error) \
  do { \
    cudaError_t err = (error); \
    if (err != cudaSuccess) { \
      printf("CUDA error: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1); \
    } \
  } while (0)

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void kernel(float *subject, float *avgs, float *stds, float *lb, float *lower_bound, float *upper_bound, int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N - M + 1) return;

  float residues = 0;
  float avg = avgs[idx];
  float std = stds[idx];

  for (int i = 0; i < M; ++i) {
    float value = (subject[idx + i] - avg) / std;
    float lower = value - lower_bound[i];
    float upper = value - upper_bound[i];

    residues += upper * upper * (upper > 0) + lower * lower * (lower < 0);
  }

  lb[idx] = residues;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: ./%s <query length> <subject length> <repeat>\n", argv[0]);
    return -1;
  }

  const int M = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  printf("Query length = %d\n", M);
  printf("Subject length = %d\n", N);

  float *subject;
  float *lower_bound;
  float *upper_bound;
  float *lb;
  float *lb_h;
  float *avgs;
  float *stds;

  CUDA_CHECK(cudaMalloc((void**)&subject, sizeof(float) * N));
  CUDA_CHECK(cudaMalloc((void**)&lower_bound, sizeof(float) * N));
  CUDA_CHECK(cudaMalloc((void**)&upper_bound, sizeof(float) * N));
  CUDA_CHECK(cudaMalloc((void**)&lb, sizeof(float) * (N - M + 1)));
  CUDA_CHECK(cudaMalloc((void**)&lb_h, sizeof(float) * (N - M + 1)));
  CUDA_CHECK(cudaMalloc((void**)&avgs, sizeof(float) * (N - M + 1)));
  CUDA_CHECK(cudaMalloc((void**)&stds, sizeof(float) * (N - M + 1)));

  srand(123);
  for (int i = 0; i < N; ++i) {
    subject[i] = (float)rand() / (float)RAND_MAX;
  }
  for (int i = 0; i < N - M + 1; ++i) {
    avgs[i] = (float)rand() / (float)RAND_MAX;
  }
  for (int i = 0; i < N - M + 1; ++i) {
    stds[i] = (float)rand() / (float)RAND_MAX;
  }
  for (int i = 0; i < M; ++i) {
    upper_bound[i] = (float)rand() / (float)RAND_MAX;
  }
  for (int i = 0; i < M; ++i) {
    lower_bound[i] = (float)rand() / (float)RAND_MAX;
  }

  const int blocks = 256;
  const int grids = (N - M + 1 + blocks - 1) / blocks;

  CUDA_CHECK(cudaMemcpy(subject, subject, sizeof(float) * N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(avgs, avgs, sizeof(float) * (N - M + 1), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(stds, stds, sizeof(float) * (N - M + 1), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(lower_bound, lower_bound, sizeof(float) * N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(upper_bound, upper_bound, sizeof(float) * N, cudaMemcpyHostToDevice));

  kernel<<<grids, blocks>>>(subject, avgs, stds, lb, lower_bound, upper_bound, M, N);
  CUDA_CHECK(cudaGetLastError());

  float *d_lb;
  CUDA_CHECK(cudaMalloc((void**)&d_lb, sizeof(float) * (N - M + 1)));
  CUDA_CHECK(cudaMemcpy(d_lb, lb, sizeof(float) * (N - M + 1), cudaMemcpyDeviceToDevice));

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    kernel<<<grids, blocks>>>(subject, avgs, stds, d_lb, lower_bound, upper_bound, M, N);
    CUDA_CHECK(cudaGetLastError());
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  CUDA_CHECK(cudaMemcpy(lb, d_lb, sizeof(float) * (N - M + 1), cudaMemcpyDeviceToHost));

  reference(subject, avgs, stds, lb_h, lower_bound, upper_bound, M, N);
  bool ok = true;
  for (int i = 0; i < N - M + 1; i++) {
    if (fabs(lb[i] - lb_h[i]) > 1e-3) {
      printf("%d %f %f\n", i, lb[i], lb_h[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  CUDA_CHECK(cudaFree(subject));
  CUDA_CHECK(cudaFree(lower_bound));
  CUDA_CHECK(cudaFree(upper_bound));
  CUDA_CHECK(cudaFree(lb));
  CUDA_CHECK(cudaFree(lb_h));
  CUDA_CHECK(cudaFree(avgs));
  CUDA_CHECK(cudaFree(stds));
  CUDA_CHECK(cudaFree(d_lb));
  return 0;
}
```
Note that the CUDA kernel launch is done using the `kernel<<<grids, blocks>>>` syntax, and the `CUDA_CHECK` macro is used to check for CUDA errors. The `cudaMemcpy` function is used to transfer data between the host and device. The `cudaMalloc` function is used to allocate memory on the device. The `cudaFree` function is used to deallocate memory on the device.